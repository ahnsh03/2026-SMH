# Traffic Sign Tuning

Run the interactive red-light tuner from Windows PowerShell:

```powershell 실행 명령어
cd C:\Users\Autonav\Desktop\SMH\src\inference
python test\tune_traffic_sign_video.py --video C:\빨간불만.mp4
python test\tune_traffic_sign_video.py --color green --video "C:\Users\Autonav\Downloads\초록불.mp4"

python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\낮_빨간불_정면.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\낮_빨간불_왼쪽.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\낮_빨간불_오른쪽.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\밤_빨간불_정면.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\밤_빨간불_왼쪽.mp4"
python test\tune_traffic_sign_video.py --video "C:\organizer_provided_videos\빨간불\밤_빨간불_오른쪽.mp4"

```

```powershell 실행 명령어
cd C:\Users\Autonav\Desktop\SMH\src\inference
python test\tune_traffic_sign_video.py --color green --video "C:\Users\Autonav\Downloads\초록불.mp4"

python test\tune_traffic_sign_video.py --color green --video "C:\organizer_provided_videos\초록불\WIN_20260617_14_39_41_Pro.mp4"
python test\tune_traffic_sign_video.py --color green --video "C:\organizer_provided_videos\초록불\WIN_20260617_14_40_20_Pro.mp4"
python test\tune_traffic_sign_video.py --color green --video "C:\organizer_provided_videos\초록불\WIN_20260617_14_41_00_Pro.mp4"
python test\tune_traffic_sign_video.py --color green --video "C:\organizer_provided_videos\초록불\밤_초록불_정면 (1).mp4"
python test\tune_traffic_sign_video.py --color green --video "C:\organizer_provided_videos\초록불\밤_초록불_왼쪽.mp4"
python test\tune_traffic_sign_video.py --color green --video "C:\organizer_provided_videos\초록불\밤_초록불_오른쪽.mp4"

```
