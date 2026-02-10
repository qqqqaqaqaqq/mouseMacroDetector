from fastapi import APIRouter
from typing import List
from macro_detector import get_macro_result, MousePoint

router = APIRouter()

# class MousePoint(BaseModel):
#     timestamp: datetime
#     x: int
#     y: int
#     deltatime: float

@router.post("/get_points")
async def get_mouse_pointer(data: List[MousePoint]):

    result = get_macro_result(data)

    if result.get("status") == "0":
        # log로 서버 콘솔에서도 확인
        data:list = result.get("data")
        
        for t in data:
            print(t["raw_error"])

    return {"status": "collecting", "buffer_count": "데이터 축적 중..."}