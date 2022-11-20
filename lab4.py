import cv2 as cv
import torch
from torchvision import transforms, models
from PIL import Image
import time

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model = torch.jit.load('Lab4/model_scripted.pt')
model.eval()
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 30)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

start_time,fps, tmp_fps = time.time(), 0, 0
while cv.waitKey(1) != 27:    
    tmp_fps += 1    
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=0.8, fy=0.8, interpolation= cv.INTER_AREA)
    
    results = model_yolo(frame)
    data = results.pandas().xyxy[0]
    data = data.loc[data['name'] == 'bowl']
    if len(data) > 0:
        batch = torch.zeros([len(data),3,224,224])
        for i in range(len(data)):
            x = data.iloc[i]
            tmp = frame[int(x['ymin']):int(x['ymax']), int(x['xmin']):int(x['xmax']),:]
            img_pil = Image.fromarray(tmp)
            img_pil = transforms(img_pil)
            batch[i] = img_pil
            
        preds = model(batch.to(device))
        preds_class = torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy()
        data['pred'] = list(map(lambda x : 'dirty' if  x > 0.6 else 'cleaned', preds_class))
        for i in range(len(data)):
            x = data.iloc[i]
            cv.rectangle(frame, (int(x['xmin']),int(x['ymin'])), (int(x['xmax']),int(x['ymax'])), (0,0,255), 2)
            cv.putText(frame, x['pred'] + '  '+ x['name'],(int(x['xmin']),int(x['ymin'])), cv.FONT_HERSHEY_SIMPLEX, 1 ,(255,255,255),2,cv.LINE_AA)
    if (time.time() - start_time) > 1:  
        start_time,fps, tmp_fps = time.time(), tmp_fps, 0  
    cv.putText(frame, str(fps) ,(20,20), cv.FONT_HERSHEY_SIMPLEX, 1 ,(255,255,255),2,cv.LINE_AA)
    
    cv.imshow('Input', frame)

cap.release()
cv.destroyAllWindows()