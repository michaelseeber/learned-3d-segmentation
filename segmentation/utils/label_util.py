import os
import sys


LABEL_COLORS = (
    ("wall", (190,153,112)),
    ("floor", (189,198,255)),
    ("cabinet", (213,255,0)), 
    ("bed", (158,0,142)),
    ("chair", (152,255,82)), 
    ("sofa", (119,77,0)), 
    ("table", (122,71,130)),
    ("door", (0,174,126)),
    ("window", (0,125,181)), 
    ("bookshelf", (0,143,156)), 
    ("picture", (107,104,130)),
    ("counter", (255,229,2)), 
    ("blinds", (117,68,177)),
    ("desk", (1,255,254)), 
    ("shelves", (0,21,68)), 
    ("curtain", (255,166,254)), 
    ("dresser", (194,140,159)), 
    ("pillow", (98,14,0)), 
    ("mirror", (0,71,84)), 
    ("floor mat", (255,219,102)), 
    ("clothes", (0,118,255)), 
    ("ceiling", (67,0,44)), 
    ("books", (1,208,255)), 
    ("refridgerator", (232,94,190)), 
    ("television", (145,208,203)), 
    ("paper", (255,147,126)), 
    ("towel", (95,173,78)), 
    ("shower curtain", (0,100,1)), 
    ("box", (255,238,232)), 
    ("whiteboard", (0,155,255)), 
    ("person", (255,0,86)), 
    ("night stand", (189,211,147)), 
    ("toilet", (133,169,0)), 
    ("sink", (149,0,58)), 
    ("lamp", (255,2,157)), 
    ("bathtub", (187,136,0)),  
    ("bag", (0,185,23)), 
    ("otherstructure", (1,0,103)),
    ("otherfurniture", (0,0,170)), 
    ("otherprop", (255,0,246)), 
    ("unknown", (0, 0, 0)), 
)

CLASSES = tuple(label for label, _ in LABEL_COLORS)
COLORS = tuple(color for _, color in LABEL_COLORS)

def label2color(label):
    return COLORS[label]

def color2label(color):
    return COLORS.index(color)
