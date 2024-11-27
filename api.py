import os
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import torch
from huggingface_hub import snapshot_download
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from inference import (
    load_vae, 
    load_unet, 
    load_scheduler,
    load_image_to_tensor_with_resize_and_crop,
    calculate_padding,
    get_unique_filename,
    seed_everething,
    SymmetricPatchifier,
    ConditioningMethod
)
from transformers import T5EncoderModel, T5Tokenizer
import imageio
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for loaded models
pipeline = None
generator = None

# Constants from inference.py
MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

# Model precision configuration
USE_BFLOAT16 = os.getenv("USE_BFLOAT16", "true").lower() == "true"

# Download configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))

# Metadata storage configuration
METADATA_FILE = os.getenv("METADATA_FILE", "metadata.json")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))

class VideoMetadata(BaseModel):
    """Model for video metadata"""
    id: str
    filename: str
    date_created: str
    prompt: str
    generation_type: str  # 'text-to-video' or 'image-to-video'
    parameters: Dict[str, Any]
    file_path: str

def load_metadata() -> Dict[str, VideoMetadata]:
    """Load metadata from file"""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                data = json.load(f)
                return {k: VideoMetadata(**v) for k, v in data.items()}
        return {}
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}

def save_metadata(metadata: Dict[str, VideoMetadata]):
    """Save metadata to file"""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump({k: v.dict() for k, v in metadata.items()}, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")

def add_video_metadata(
    filename: str,
    prompt: str,
    generation_type: str,
    parameters: Dict[str, Any],
    file_path: str
) -> str:
    """Add new video metadata and return video ID"""
    metadata = load_metadata()
    video_id = f"{len(metadata):06d}"
    
    metadata[video_id] = VideoMetadata(
        id=video_id,
        filename=filename,
        date_created=datetime.now().isoformat(),
        prompt=prompt,
        generation_type=generation_type,
        parameters=parameters,
        file_path=file_path
    )
    
    save_metadata(metadata)
    return video_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, generator
    
    logger.info("Starting LTX Video API server...")
    
    try:
        # Get current working directory and create relative paths
        current_dir = Path.cwd()
        ckpt_dir = current_dir / os.getenv("CKPT_DIR", "model").lstrip("./")
        
        # Download model if not exists
        if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
            logger.info(f"Model not found in {ckpt_dir}. Downloading with {MAX_WORKERS} workers...")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                "Lightricks/LTX-Video",
                local_dir=str(ckpt_dir),
                local_dir_use_symlinks=False,
                repo_type='model',
                max_workers=MAX_WORKERS,
                resume_download=True,
                etag_timeout=30
            )
            logger.info("Model download completed successfully")
        
        unet_dir = ckpt_dir / "unet"
        vae_dir = ckpt_dir / "vae"
        scheduler_dir = ckpt_dir / "scheduler"

        logger.info("Loading VAE model...")
        vae = load_vae(vae_dir)
        
        logger.info("Loading UNet model...")
        unet = load_unet(unet_dir)
        if USE_BFLOAT16 and unet.dtype != torch.bfloat16:
            logger.info("Converting UNet to bfloat16...")
            unet = unet.to(torch.bfloat16)
        
        logger.info("Loading scheduler...")
        scheduler = load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)
        
        logger.info("Loading text encoder and tokenizer...")
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", 
            subfolder="text_encoder"
        )
        if torch.cuda.is_available():
            text_encoder = text_encoder.to("cuda")
            
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            subfolder="tokenizer"
        )

        pipeline = LTXVideoPipeline(
            transformer=unet,
            patchifier=patchifier,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae=vae
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            logger.info("Pipeline moved to CUDA")
            
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Server startup complete!")
        
        yield
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")
    finally:
        pipeline = None
        generator = None

app = FastAPI(
    title="LTX Video Generation API",
    description="API for generating videos using LTX-Video model",
    version="1.0.0",
    lifespan=lifespan
)

class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(
        default=os.getenv("DEFAULT_NEGATIVE_PROMPT"),
        description="Negative prompt for undesired features"
    )
    num_inference_steps: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", 40)),
        description="Number of inference steps"
    )
    guidance_scale: Optional[float] = Field(
        default=float(os.getenv("DEFAULT_GUIDANCE_SCALE", 3.0)),
        description="Guidance scale for the pipeline"
    )
    height: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_HEIGHT", 480)),
        description="Height of the output video frames"
    )
    width: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_WIDTH", 704)), 
        description="Width of the output video frames"
    )
    num_frames: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_NUM_FRAMES", 121)),
        description="Number of frames to generate"
    )
    frame_rate: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_FRAME_RATE", 25)),
        description="Frame rate for the output video"
    )
    seed: Optional[int] = Field(
        default=171198,
        description="Random seed for generation"
    )
    use_mixed_precision: Optional[bool] = Field(
        default=not USE_BFLOAT16,
        description="Whether to use mixed precision during inference"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A clear, turquoise river flows through a rocky canyon",
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "num_inference_steps": 40,
                "guidance_scale": 3.0,
                "height": 480,
                "width": 704,
                "num_frames": 121,
                "frame_rate": 25,
                "seed": 171198,
                "use_mixed_precision": False
            }
        }

def validate_dimensions(height: int, width: int, num_frames: int):
    """Validate input dimensions against maximum allowed values"""
    if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Input dimensions {height}x{width}x{num_frames} exceed maximum allowed values ({MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES})"
        )

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if the service is healthy and models are loaded"""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Models not loaded"}
        )
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "using_bfloat16": USE_BFLOAT16
    }

@app.get("/videos")
async def list_videos(
    date: Optional[str] = None,
    generation_type: Optional[str] = None
) -> List[VideoMetadata]:
    """List all generated videos with optional date and type filters"""
    metadata = load_metadata()
    videos = list(metadata.values())
    
    if date:
        videos = [v for v in videos if v.date_created.startswith(date)]
    if generation_type:
        videos = [v for v in videos if v.generation_type == generation_type]
        
    return videos

@app.get("/videos/{video_id}")
async def get_video(video_id: str) -> VideoMetadata:
    """Get metadata for a specific video"""
    metadata = load_metadata()
    if video_id not in metadata:
        raise HTTPException(status_code=404, detail="Video not found")
    return metadata[video_id]

@app.get("/videos/{video_id}/download")
async def download_video(video_id: str):
    """Download a specific video"""
    metadata = load_metadata()
    if video_id not in metadata:
        raise HTTPException(status_code=404, detail="Video not found")
        
    video_path = metadata[video_id].file_path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
        
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=metadata[video_id].filename
    )

@app.post("/generate/text-to-video")
async def generate_text_to_video(params: GenerationParams):
    """Generate video from text prompt"""
    try:
        logger.info(f"Starting text-to-video generation with prompt: {params.prompt}")
        
        validate_dimensions(params.height, params.width, params.num_frames)
        
        seed_everething(params.seed)
        generator.manual_seed(params.seed)
        
        height_padded = ((params.height - 1) // 32 + 1) * 32
        width_padded = ((params.width - 1) // 32 + 1) * 32
        num_frames_padded = ((params.num_frames - 2) // 8 + 1) * 8 + 1
        
        logger.info(f"Using padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}")
        
        sample = {
            "prompt": params.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": params.negative_prompt,
            "negative_prompt_attention_mask": None,
            "media_items": None,
        }

        logger.info("Generating video...")
        images = pipeline(
            num_inference_steps=params.num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=params.guidance_scale,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=params.frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=ConditioningMethod.UNCONDITIONAL,
            mixed_precision=params.use_mixed_precision,
        ).images

        output_dir = OUTPUT_DIR / datetime.today().strftime('%Y-%m-%d')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        output_filename = get_unique_filename(
            "text_to_vid_0",
            ".mp4",
            prompt=params.prompt,
            seed=params.seed,
            resolution=(params.height, params.width, params.num_frames),
            dir=output_dir
        )

        logger.info(f"Saving video to {output_filename}")
        with imageio.get_writer(output_filename, fps=params.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)
                
        # Add metadata
        video_id = add_video_metadata(
            filename=output_filename.name,
            prompt=params.prompt,
            generation_type="text-to-video",
            parameters=params.dict(),
            file_path=str(output_filename)
        )
                
        return FileResponse(
            output_filename,
            media_type="video/mp4",
            filename=output_filename.name,
            headers={"X-Video-ID": video_id}
        )

    except Exception as e:
        logger.error(f"Error in text-to-video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/image-to-video")
async def generate_image_to_video(
    file: UploadFile = File(...),
    params: GenerationParams = None
):
    """Generate video from input image and text prompt"""
    try:
        logger.info(f"Starting image-to-video generation with prompt: {params.prompt}")
        
        validate_dimensions(params.height, params.width, params.num_frames)
        
        temp_image_path = f"temp_{file.filename}"
        try:
            with open(temp_image_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info("Processing input image...")
            media_items_prepad = load_image_to_tensor_with_resize_and_crop(
                temp_image_path, 
                params.height, 
                params.width
            )
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        
        height_padded = ((params.height - 1) // 32 + 1) * 32
        width_padded = ((params.width - 1) // 32 + 1) * 32
        num_frames_padded = ((params.num_frames - 2) // 8 + 1) * 8 + 1
        
        logger.info(f"Using padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}")
        
        padding = calculate_padding(params.height, params.width, height_padded, width_padded)
        
        media_items = torch.nn.functional.pad(
            media_items_prepad, 
            padding, 
            mode="constant", 
            value=-1
        )

        seed_everething(params.seed)
        generator.manual_seed(params.seed)
        
        sample = {
            "prompt": params.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": params.negative_prompt,
            "negative_prompt_attention_mask": None,
            "media_items": media_items,
        }

        logger.info("Generating video...")
        images = pipeline(
            num_inference_steps=params.num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=params.guidance_scale,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=params.frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=ConditioningMethod.FIRST_FRAME,
            mixed_precision=params.use_mixed_precision,
        ).images

        output_dir = OUTPUT_DIR / datetime.today().strftime('%Y-%m-%d')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        output_filename = get_unique_filename(
            "img_to_vid_0",
            ".mp4",
            prompt=params.prompt,
            seed=params.seed,
            resolution=(params.height, params.width, params.num_frames),
            dir=output_dir
        )

        logger.info(f"Saving video to {output_filename}")
        with imageio.get_writer(output_filename, fps=params.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)
                
        # Add metadata
        video_id = add_video_metadata(
            filename=output_filename.name,
            prompt=params.prompt,
            generation_type="image-to-video",
            parameters=params.dict(),
            file_path=str(output_filename)
        )
                
        return FileResponse(
            output_filename,
            media_type="video/mp4",
            filename=output_filename.name,
            headers={"X-Video-ID": video_id}
        )

    except Exception as e:
        logger.error(f"Error in image-to-video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
