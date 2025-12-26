# ProjectSegmentAgent-CVLab

cd Mock_pipeline
run main.py

/
bấm chuột trái để tạo điểm neo mới, bấm chuột phải để kết thúc. 


/
bấm ESC để chuyển từ segment -> inpainting -> save outputs

Tóm tắt lý do:
    - Do intelligent_scissors cắt mask quá sát
    - Deeplearning inpainting bản chất là phải hiểu semantic của object
Cụ thể:
    Khi dùng intelligent_scissors, thì cái mask do nó tạo ra rất sát so với object, nên đôi lúc mask bị thiếu mất vài pixel, do đó lúc mà mình tô thủ công quả bóng thành vòng tròn, thì các pixel mà mask không bao được kia, nó tạo thành 1 vòng tròn pixel, mà do Deeplearning inpainting hiểu theo semantic, nên nó sẽ hiểu là có một quả bóng hình tròn ở đó, nên nó inpaint thành hình tròn luôn (còn màu nâu thì là do các pixel đó ở biên nên thực chất nó có màu nâu chứ không phải đỏ). Chứng minh suy luận trên đúng bằng cách ko cắt mask quá sát object, chỉ cho một vài chỗ là sát, thì sẽ hiểu chứng minh trên.
