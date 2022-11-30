import cv2 as cv
import torch
from torchvision import transforms
from PIL import Image
import time

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True).to(device)
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

# @profile
# def foo():
start_time, fps, tmp_fps = time.time(), 0, 0
while cv.waitKey(1) != 27:    #press Esc
    tmp_fps += 1    
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=0.8, fy=0.8, interpolation= cv.INTER_AREA)
    
    results = model_yolo(frame)
    data = list(filter(lambda x: x['cls'] == 45, results.crop(save=False)))
    if len(data) > 0:
        batch = torch.zeros([len(data),3,224,224])
        for i in range(len(data)):
            batch[i] = transforms(Image.fromarray(data[i]['im']))
        preds = model(batch.to(device))
        preds_class = torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy() 
        
        for temp_frame, pred in zip(data, preds_class):
            x = temp_frame['box']
            cv.rectangle(frame, (int(x[0]),int(x[1])), (int(x[2]),int(x[3])), (0,0,255), 2)
            line = ''
            line += ('dirty' if pred > 0.6 else 'cleaned') + ' ' + str(round(pred,2))
            # line += '  '+ temp_frame['label']
            cv.putText(frame, line, (int(x[0]),int(x[1])), cv.FONT_HERSHEY_SIMPLEX, 1 ,(255,255,255),2,cv.LINE_AA)
    
    if (time.time() - start_time) > 1:  
        start_time, fps, tmp_fps = time.time(), tmp_fps, 0  
    cv.putText(frame, str(fps) ,(20,20), cv.FONT_HERSHEY_SIMPLEX, 1 ,(255,255,255),2,cv.LINE_AA)
    
    cv.imshow('Input', frame)

# foo()

cap.release()
cv.destroyAllWindows()