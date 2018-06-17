NYU40_LABEL_COLORS = (
    ("wall", (176, 135, 93)), # #B0875D
    ("floor", (175, 183, 255)), # #AFB7FF
    ("cabinet", (205, 255, 10)), # #CDFF0A
    ("bed", (138, 0, 123)), # #8A007B
    ("chair", (139, 255, 65)), # #8BFF41
    ("sofa", (98, 60, 1)), # #623C01
    ("table", (101, 51, 111)), # #65336F
    ("door", (20, 161, 106)), # #14A16A
    ("window", (13, 104, 166)), # #0D68A6
    ("bookshelf", (15, 125, 138)), # #0F7D8A
    ("picture", (88, 84, 110)), # #58546E
    ("counter", (254, 226, 10)), # #FEE20A
    ("blinds", (97, 44, 162)), # #612CA2
    ("desk", (33, 255, 253)), # #21FFFD
    ("shelves", (0, 14, 52)), # #000E34
    ("curtain", (253, 142, 255)), # #FD8EFF
    ("dresser", (179, 119, 141)), # #B3778D
    ("pillow", (78, 8, 0)), # #4E0800
    ("mirror", (6, 55, 66)), # #063742
    ("floor mat", (254, 213, 83)), # #FED553
    ("clothes", (9, 90, 254)), # #095AFE
    ("ceiling", (51, 0, 33)), # #330021
    ("books", (25, 198, 255)), # #19C6FF
    ("refridgerator", (224, 64, 176)), # #E040B0
    ("television", (128, 199, 191)), # #80C7BF
    ("paper", (253, 126, 108)), # #FD7E6C
    ("towel", (80, 160, 61)), # #50A03D
    ("shower curtain", (11, 84, 2)), # #0B5402
    ("box", (255, 234, 226)), # #FFEAE2
    ("whiteboard", (16, 133, 255)), # #1085FF
    ("person", (252, 0, 69)), # #FC0045
    ("night stand", (176, 204, 128)), # #B0CC80
    ("toilet", (115, 157, 4)), # #739D04
    ("sink", (129, 0, 44)), # #81002C
    ("lamp", (251, 0, 140)), # #FB008C
    ("bathtub", (172, 118, 5)), # #AC7605
    ("bag", (23, 176, 19)), # #17B013
    ("otherstructure", (0, 0, 83)), # #000053
    ("otherfurniture", (0, 0, 255)), # #0000FF
    ("otherprop", (251, 0, 244)), # #FB00F4
    ("unknown", (0, 0, 0)), # #000000
)

NYU40_LABELS = tuple(label for label, _ in NYU40_LABEL_COLORS)
NYU40_COLORS = tuple(color for _, color in NYU40_LABEL_COLORS)