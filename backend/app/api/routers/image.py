from fastapi import APIRouter
router = APIRouter(
    prefix="/image",
    tags = ["image"]
)

@router.get("/")
async def read_images():
    return [{"image_title":"image_1"},{"image_title":"image_2"}]

@router.get("/{image_id}")
async def read_image(image_id:str):
    return {"image_id":image_id}