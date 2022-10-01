from PIL import Image,ImageDraw,ImageFont
import os
import json
from tqdm import tqdm
# 이미지로 출력할 글자 및 폰트 지정 
draw_text = 'WARMING'
font = ImageFont.truetype("./gray_text/font/VerilySerifMono.ttf", 64)
 
# 이미지 사이즈 지정
text_width = 256
text_height = 64
 
# 이미지 객체 생성 (배경 검정)
canvas = Image.new('RGB', (text_width, text_height), "white")
 
# 가운데에 그리기 (폰트 색: 하양)
draw = ImageDraw.Draw(canvas)
w, h = font.getsize(draw_text)
draw.text(((text_width-w)/2.0,(text_height-h)/2.0), draw_text, 'black', font)
 
# png로 저장 및 출력해서 보기
canvas.save(draw_text+'.png', "PNG")



def get_canvas_from_text(font_path, text, size:tuple):
    font = ImageFont.truetype(font_path, size[1])
    text_width, text_height = size

    
    if len(text) < 7:
        canvas = Image.new('RGB', (text_width, text_height), "white")
        draw = ImageDraw.Draw(canvas)
        w, h = font.getsize(text)
        draw.text(((text_width-w)/2.0,(text_height-h)/2.0), text, 'black', font)
    elif len(text) >= 7:
        w, h = font.getsize(text)
        canvas = Image.new('RGB', (w, text_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((0,(text_height-h)/2.0), text, 'black', font)
        canvas = canvas.resize((text_width, text_height))

    return canvas




if __name__ == "__main__":

    label_path = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/label_dic.json"
    font_path = "/home/sy/textailor_CLAB/gray_text/font/VerilySerifMono.ttf"
    output_path = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/gray_text"
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    assert os.path.exists(label_path), "label_dic.json does not exist"
    assert os.path.exists(font_path), "font does not exist"

    print('loading label_dic.json')

    with open(label_path, "r") as f:
        label_dic = json.load(f)

    pbar = tqdm(list(label_dic.items()))
    
    print('start rendering')
    for key, value in pbar:
        canvas = get_canvas_from_text(font_path, value, (256, 64))
        canvas.save(os.path.join(output_path, key+".png"), "PNG")

        pbar.set_description("Processing %s" % key)

    print('rendering done')
    # test_texts = ["A", "AB", "ABC", "ABCD", "ABCDE", "ABCDEF", "ABCDEFG", "ABCDEFGH","ABCDEFGHIJK"]
    # # test_texts = ["ABCDEFGHIJK"]
    # for text in test_texts:
    #     lower_text = text.lower()

    #     canvas = get_canvas_from_text(font_path, text, (256, 64))
    #     lower_canvas = get_canvas_from_text(font_path, lower_text, (256, 64))

    #     canvas.save(os.path.join("./gray_text/result",text+".png"), "PNG")
    #     lower_canvas.save(os.path.join("./gray_text/result",lower_text+".png"), "PNG")




    # canvas = get_canvas_from_text("./gray_text/font/VerilySerifMono.ttf", "WARMING", (256,64))
    # canvas.save("WARMING.png", "PNG")




