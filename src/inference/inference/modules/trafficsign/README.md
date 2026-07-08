# Traffic Sign Tuning

Run the interactive red-light tuner from Windows PowerShell:

```powershell 실행 명령어
cd C:\Users\Autonav\Desktop\SMH\src\inference
python test\tune_traffic_sign_video.py --video C:\빨간불만.mp4

python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\낮_빨간불_정면.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\낮_빨간불_왼쪽.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\낮_빨간불_오른쪽.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\밤_빨간불_정면.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\밤_빨간불_왼쪽.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\밤_빨간불_오른쪽.mp4"

```