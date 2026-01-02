from fastapi import FastAPI
from app.api.routers import image

app = FastAPI()

app.include_router(image.router)
@app.get("/")
async def root():
    return {"message": "Image API System"}

