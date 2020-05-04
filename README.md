## FACE Filters with Flask and Open-CV
```python
pip install -r requirements.txt
```
### Run Server
```python
python app.py
```
#### Use Built-in Webcam of Laptop
##### Put Zero (O) in cv2.VideoCapture(0)
```python
cv2.VideoCapture(0)
```
#### Use Ip Camera/CCTV/RTSP Link
```python
cv2.VideoCapture('rtsp://username:password@camera_ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp')  
 ```
 
#### Display the resulting frame in browser
```python
cv2.imencode('.jpg', frame)[1].tobytes()                 
``` 
## Or this one

 ```python
net , buffer = cv2.imencode('.jpg', frame)
buffer.tobytes()              
```   

