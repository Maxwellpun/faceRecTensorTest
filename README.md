# faceRecTensorTest ขั้นตอน
1. download model file:
and put it in folder named "model"
เป็นโมเดลตรวจจับใบหน้า

2. create folder name "train_img" and "aligned_img"
put sample pictures inside "train_img" in saparate folder for each person.
จะตัดแค่ส่วนใบหน้าจากภาพขนาดต่างๆมาเป็นไฟล์ขนาดเท่าๆกัน เพื่อนำไป train ต่อใน train_img
ขั้นจอนนี้ก็จะมีโฟลเดอร์ npy เป็นของ mtcnn ไว้ว่ากันครับอันนี้

3. run data_preprocess.py and then run train_main.py
it will create classes in class folder and the sample pictures are no longer need.
