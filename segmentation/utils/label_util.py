import os
import sys

LABEL_COLORS = (
    ("unknown", (0, 0, 0)), 
    ("floor", (189,198,255)),
    ("wall", (190,153,112)),
    ("cabinet", (213,255,0)), 
    ("bed", (158,0,142)),
    ("chair", (152,255,82)),
    ("sofa", (119,77,0)), 
    ("table", (122,71,130)), 
    ("door", (0,174,126)),
    ("window", (0,125,181)), 
    ("refridgerator", (232,94,190)), 
    ("bookshelf", (0,143,156)), 
    ("picture", (107,104,130)),
    ("counter", (255,229,2)), 
    ("desk", (1,255,254)), 
    ("curtain", (255,166,254)), 
    ("shower curtain", (0,100,1)), 
    ("bathtub", (187,136,0)),  
    ("floor mat", (255,219,102)),
    ("toilet", (133,169,0)), 
    ("sink", (149,0,58)), 
    ("clothes", (0,118,255)), 
    ("blinds", (117,68,177)),
    ("shelves", (0,21,68)), 
    ("dresser", (194,140,159)), 
    ("pillow", (98,14,0)), 
    ("mirror", (0,71,84)), 
    ("ceiling", (67,0,44)), 
    ("books", (1,208,255)), 
    ("television", (145,208,203)), 
    ("paper", (255,147,126)), 
    ("towel", (95,173,78)), 
    ("box", (255,238,232)), 
    ("whiteboard", (0,155,255)), 
    ("person", (255,0,86)), 
    ("night stand", (189,211,147)), 
    ("lamp", (255,2,157)), 
    ("bag", (0,185,23)), 
    ("otherfurniture", (0,0,255)), 
    ("otherprop", (255,0,246)), 
    ("otherstructure", (1,0,103)),
)

CLASSES = tuple(label for label, _ in LABEL_COLORS)
COLORS = tuple(color for _, color in LABEL_COLORS)

def label2color(label):
        return COLORS[label]

def label2class(label):
    return CLASSES[label]

def color2label(color):
    return COLORS.index(color)

