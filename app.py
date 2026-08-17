import os
import uuid
import numpy as np
import torch

from flask import Flask, render_template, request, send_from_directory
from PIL import Image

from model import create_model


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\anish\OneDrive\Desktop\Semicon"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pth"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# Create folders
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = create_model()
model = model.to(device)

if os.path.exists(MODEL_PATH):

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print("Model loaded successfully.")

else:

    print("ERROR: Model file not found:")
    print(MODEL_PATH)


# ============================================================
# LOAD INPUT IMAGE
# ============================================================

def load_input_image(filepath):

    extension = os.path.splitext(filepath)[1].lower()

    # --------------------------------------------------------
    # NPY
    # --------------------------------------------------------

    if extension == ".npy":

        image = np.load(filepath)

        image = image.astype(np.float32)

        print("Original NPY shape:", image.shape)

        # Normalize
        image = image - image.min()

        if image.max() > 0:
            image = image / image.max()

        # ----------------------------------------------------
        # 2D IMAGE
        # ----------------------------------------------------

        if image.ndim == 2:

            image = np.expand_dims(
                image,
                axis=0
            )

        # ----------------------------------------------------
        # 3D IMAGE
        # ----------------------------------------------------

        elif image.ndim == 3:

            # H,W,C -> C,H,W
            if image.shape[-1] in [1, 3]:

                image = np.transpose(
                    image,
                    (2, 0, 1)
                )

        else:

            raise ValueError(
                "Unsupported NPY shape: "
                + str(image.shape)
            )

        tensor = torch.from_numpy(
            image
        ).float()

    # --------------------------------------------------------
    # PNG / JPG / JPEG
    # --------------------------------------------------------

    elif extension in [".png", ".jpg", ".jpeg"]:

        image = Image.open(
            filepath
        ).convert("L")

        image = np.array(
            image
        ).astype(
            np.float32
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        tensor = torch.from_numpy(
            image
        ).float()

    else:

        raise ValueError(
            "Unsupported file type: "
            + extension
        )

    return tensor


# ============================================================
# CREATE BEFORE IMAGE
# ============================================================

def create_before_image(
    filepath,
    before_path
):

    extension = os.path.splitext(
        filepath
    )[1].lower()

    # --------------------------------------------------------
    # NPY
    # --------------------------------------------------------

    if extension == ".npy":

        image = np.load(
            filepath
        )

        image = image.astype(
            np.float32
        )

        # Normalize
        image = image - image.min()

        if image.max() > 0:

            image = (
                image /
                image.max()
            )

        # ----------------------------------------------------
        # Convert dimensions
        # ----------------------------------------------------

        if image.ndim == 3:

            # H,W,C
            if image.shape[-1] in [1, 3]:

                image = np.transpose(
                    image,
                    (2, 0, 1)
                )

            # C,H,W
            image = np.squeeze(
                image
            )

        # ----------------------------------------------------
        # Convert to 8-bit
        # ----------------------------------------------------

        image = (
            image * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        # Save as PNG
        Image.fromarray(
            image
        ).save(
            before_path
        )

    # --------------------------------------------------------
    # NORMAL IMAGE
    # --------------------------------------------------------

    else:

        image = Image.open(
            filepath
        )

        image.save(
            before_path
        )


# ============================================================
# SAVE RESTORED IMAGE
# ============================================================

def save_output(
    output,
    original_filename
):

    output = output.detach().cpu().numpy()

    output = np.squeeze(
        output
    )

    # Normalize
    output = output - output.min()

    if output.max() > 0:

        output = (
            output /
            output.max()
        )

    # Convert to PNG
    output = (
        output * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # Always save as PNG
    base_name = os.path.splitext(
        original_filename
    )[0]

    output_filename = (
        "restored_"
        + base_name
        + ".png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    Image.fromarray(
        output
    ).save(
        output_path
    )

    return output_filename


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    # ========================================================
    # GET
    # ========================================================

    if request.method == "GET":

        return render_template(
            "index.html"
        )


    # ========================================================
    # POST
    # ========================================================

    if "file" not in request.files:

        return render_template(
            "index.html",
            error="Please choose a file."
        )


    file = request.files["file"]


    if file.filename == "":

        return render_template(
            "index.html",
            error="Please choose a file."
        )


    # ========================================================
    # CHECK FILE TYPE
    # ========================================================

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    allowed_extensions = [
        ".npy",
        ".png",
        ".jpg",
        ".jpeg"
    ]

    if extension not in allowed_extensions:

        return render_template(
            "index.html",
            error="Please upload NPY, PNG, JPG or JPEG."
        )


    # ========================================================
    # SAVE UPLOADED FILE
    # ========================================================

    unique_id = str(
        uuid.uuid4()
    )

    uploaded_filename = (
        unique_id
        + extension
    )

    uploaded_path = os.path.join(
        UPLOAD_DIR,
        uploaded_filename
    )

    file.save(
        uploaded_path
    )

    print()
    print("Uploaded file:")
    print(uploaded_path)


    # ========================================================
    # RESTORE IMAGE
    # ========================================================

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        input_tensor = load_input_image(
            uploaded_path
        )

        print(
            "Input tensor shape:",
            input_tensor.shape
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        input_tensor = input_tensor.unsqueeze(
            0
        )

        input_tensor = input_tensor.to(
            device
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_tensor
            )


        print(
            "Output shape:",
            output.shape
        )


        # ====================================================
        # SAVE RESTORED IMAGE
        # ====================================================

        output_filename = save_output(
            output,
            file.filename
        )


        # ====================================================
        # CREATE BEFORE IMAGE
        # ====================================================

        before_filename = (
            "before_"
            + unique_id
            + ".png"
        )

        before_path = os.path.join(
            UPLOAD_DIR,
            before_filename
        )


        create_before_image(
            uploaded_path,
            before_path
        )


        # ====================================================
        # BROWSER URLS
        # ====================================================

        before_url = (
            "/uploads/"
            + before_filename
        )

        after_url = (
            "/outputs/"
            + output_filename
        )


        print()
        print("Restoration completed.")
        print("Before:", before_url)
        print("After :", after_url)
        print()


        # ====================================================
        # SHOW RESULT
        # ====================================================

        return render_template(
            "index.html",
            before=before_url,
            after=after_url
        )


    # ========================================================
    # ERROR
    # ========================================================

    except Exception as e:

        print()
        print("ERROR:")
        print(e)
        print()

        return render_template(
            "index.html",
            error=str(e)
        )


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# ============================================================
# SERVE RESTORED IMAGES
# ============================================================

@app.route(
    "/outputs/<filename>"
)
def output_file(filename):

    return send_from_directory(
        OUTPUT_DIR,
        filename
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("   AI IMAGE RESTORATION APPLICATION")
    print("======================================")
    print()
    print("Open this address:")
    print()
    print("http://127.0.0.1:5000")
    print()

    app.run(
        debug=True
    )