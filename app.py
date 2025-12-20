import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time




st.set_page_config(
    page_title="Vision Blaze+",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Vision Blaze+ : Interactive Computer Vision Analysis Platform")




if "image" not in st.session_state:
    st.session_state.image = None

if "log" not in st.session_state:
    st.session_state.log = []




def log_operation(name):
    st.session_state.log.append({
        "operation": name,
        "timestamp": time.strftime("%H:%M:%S")
    })

def load_image(file):
    return Image.open(file).convert("RGB")

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))




def image_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-7))

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))




menu = st.sidebar.radio(
    "🔬 Vision Operations",
    [
        "Upload Image",
        "Edge Analysis",
        "Contrast Enhancement",
        "Saliency Mapping",
        "Segmentation",
        "Operation Log"
    ]
)




if menu == "Upload Image":
    file = st.file_uploader("📤 Upload Image", ["png", "jpg", "jpeg"])
    if file:
        st.session_state.image = load_image(file)
        log_operation("Image Uploaded")
        st.success("Image loaded successfully")
        st.image(st.session_state.image, use_container_width=True)




elif menu == "Edge Analysis" and st.session_state.image:
    img = pil_to_cv(st.session_state.image)
    t1 = st.slider("Lower Threshold", 0, 255, 80)
    t2 = st.slider("Upper Threshold", 0, 255, 180)

    edges = cv2.Canny(img, t1, t2)
    log_operation("Edge Detection (Canny)")

    col1, col2 = st.columns(2)
    col1.image(st.session_state.image, caption="Original", use_container_width=True)
    col2.image(edges, caption="Edges", use_container_width=True)




elif menu == "Contrast Enhancement" and st.session_state.image:
    img = pil_to_cv(st.session_state.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    log_operation("Histogram Equalization")

    entropy_before = image_entropy(img)
    entropy_after = image_entropy(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    st.metric("Entropy Before", round(entropy_before, 3))
    st.metric("Entropy After", round(entropy_after, 3))

    st.image(enhanced, caption="Enhanced Image", use_container_width=True)




elif menu == "Saliency Mapping" and st.session_state.image:
    img = pil_to_cv(st.session_state.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    saliency = cv2.magnitude(sobelx, sobely)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    log_operation("Saliency Mapping")

    st.image(saliency, caption="Saliency Map", use_container_width=True)




elif menu == "Segmentation" and st.session_state.image:
    img = pil_to_cv(st.session_state.image)
    mask = np.zeros(img.shape[:2], np.uint8)
    bg, fg = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)

    h, w = img.shape[:2]
    rect = (20, 20, w-40, h-40)

    cv2.grabCut(img, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    segmented = img * mask2[:,:,np.newaxis]

    log_operation("GrabCut Segmentation")

    st.image(cv_to_pil(segmented), caption="Segmented Output", use_container_width=True)



elif menu == "Operation Log":
    st.subheader("🧾 Experiment Log")
    if st.session_state.log:
        st.table(st.session_state.log)
    else:
        st.info("No operations recorded yet")




elif not st.session_state.image:
    st.info("⬅ Upload an image to begin")
