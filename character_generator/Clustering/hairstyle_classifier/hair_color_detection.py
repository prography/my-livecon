from face_classifier.skin_color_detection import extractDominantColor

def hair_color_main(hair_masked):
    print("input:", hair_masked.shape)
    colors = extractDominantColor(hair_masked)
    hair_color = tuple(map(int, (colors[1]['color'])))
    return hair_color