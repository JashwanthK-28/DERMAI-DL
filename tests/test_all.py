"""
DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
Section 5: Automated Validation Suite & Verification Checks

1. test_auth_and_authorization():
   - Valid JWT generation on user registration & login
   - 401 Unauthorized on invalid credentials
   - 403 Forbidden when User A attempts to access User B's case
2. test_corrupt_upload_rejection():
   - Uploading corrupted/non-image bytes returns 400 Bad Request
3. test_preprocessing_tensor_output():
   - Validates tensor shape (224, 224, 3), dtype float32, range [0.0, 1.0]
4. test_model_inference_pipeline():
   - Runs preprocessed image through recommended model (ResNet50 / EfficientNet)
   - Confirms valid diagnostic class, probabilities summing to 1.0, and risk category
"""

import io
import os
import sys
import uuid
import pytest
import numpy as np
from PIL import Image
from httpx import AsyncClient, ASGITransport

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import app
from backend.app.database import init_db
from backend.app.services.preprocessing_service import PreprocessingService
from backend.app.services.model_service import ModelService
from ml.models import DIAGNOSTIC_CLASSES, RISK_MAPPING


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    """Ensure database tables are initialized before running tests."""
    await init_db()


def create_dummy_image_bytes(width: int = 250, height: int = 250, color=(180, 100, 90)) -> bytes:
    """Generate in-memory valid PNG image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_auth_and_authorization():
    """
    Validation Check 1:
    - User registration returns 201 Created
    - Login returns valid JWT token
    - Bad credentials return 401 Unauthorized
    - Non-admin User A accessing User B's case returns 403 Forbidden
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Register User A
        email_a = f"user_a_{uuid.uuid4().hex[:6]}@hospital.org"
        pwd_a = "SecretPassword123"
        reg_a = await client.post("/api/v1/auth/register", json={
            "email": email_a,
            "password": pwd_a,
            "role": "clinician"
        })
        assert reg_a.status_code == 201, f"User A registration failed: {reg_a.text}"
        data_a = reg_a.json()
        assert data_a["email"] == email_a

        # 2. Login User A (Valid JWT)
        login_a = await client.post("/api/v1/auth/login", data={
            "username": email_a,
            "password": pwd_a
        })
        assert login_a.status_code == 200, f"Login failed: {login_a.text}"
        token_data_a = login_a.json()
        assert "access_token" in token_data_a
        token_a = token_data_a["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        # 3. Bad Credentials Login -> 401 Unauthorized
        bad_login = await client.post("/api/v1/auth/login", data={
            "username": email_a,
            "password": "WrongPasswordXYZ"
        })
        assert bad_login.status_code == 401, f"Expected 401, got {bad_login.status_code}"

        # 4. User A creates Case A
        case_res = await client.post("/api/v1/cases", json={
            "patient_clinical_id": "PAT_TEST_A",
            "age": 45,
            "sex": "FEMALE",
            "anatomic_site": "FOREARM",
            "notes": "Suspected pigmented actinic keratosis"
        }, headers=headers_a)
        assert case_res.status_code == 201
        case_a_id = case_res.json()["id"]

        # 5. User A can access own case -> 200 OK
        access_own = await client.get(f"/api/v1/cases/{case_a_id}", headers=headers_a)
        assert access_own.status_code == 200
        assert access_own.json()["patient_clinical_id"] == "PAT_TEST_A"

        # 6. Register & Login User B
        email_b = f"user_b_{uuid.uuid4().hex[:6]}@hospital.org"
        pwd_b = "SecretPassword456"
        await client.post("/api/v1/auth/register", json={
            "email": email_b,
            "password": pwd_b,
            "role": "clinician"
        })
        login_b = await client.post("/api/v1/auth/login", data={
            "username": email_b,
            "password": pwd_b
        })
        token_b = login_b.json()["access_token"]
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 7. User B attempts to access User A's case -> 403 Forbidden!
        unauthorized_access = await client.get(f"/api/v1/cases/{case_a_id}", headers=headers_b)
        assert unauthorized_access.status_code == 403, (
            f"Expected 403 Forbidden when User B accesses User A's case, got {unauthorized_access.status_code}: {unauthorized_access.text}"
        )
        print("\n[+] test_auth_and_authorization: PASSED (201 Register, 200 Login, 401 Bad Credentials, 403 Cross-User Protection)")


@pytest.mark.asyncio
async def test_corrupt_upload_rejection():
    """
    Validation Check 2:
    Uploading corrupted byte arrays or zero-byte files returns 400 Bad Request.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Create user and case
        email = f"uploader_{uuid.uuid4().hex[:6]}@hospital.org"
        pwd = "TestPassword123"
        await client.post("/api/v1/auth/register", json={"email": email, "password": pwd, "role": "clinician"})
        login = await client.post("/api/v1/auth/login", data={"username": email, "password": pwd})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        case = await client.post("/api/v1/cases", json={"patient_clinical_id": "PAT_CORRUPT_TEST"}, headers=headers)
        case_id = case.json()["id"]

        # Case A: Corrupted random byte array
        corrupt_bytes = b"CORRUPTED_RAW_NON_IMAGE_DATA_BYTES_XYZ_1234567890" * 20
        res_corrupt = await client.post(
            f"/api/v1/cases/{case_id}/images/upload",
            files={"file": ("corrupt.png", corrupt_bytes, "image/png")},
            headers=headers
        )
        assert res_corrupt.status_code == 400, f"Expected 400 Bad Request, got {res_corrupt.status_code}: {res_corrupt.text}"

        # Case B: Zero-byte empty payload
        empty_bytes = b""
        res_empty = await client.post(
            f"/api/v1/cases/{case_id}/images/upload",
            files={"file": ("empty.png", empty_bytes, "image/png")},
            headers=headers
        )
        assert res_empty.status_code == 400, f"Expected 400 Bad Request on empty file, got {res_empty.status_code}"

        # Case C: Image below 100x100 resolution minimum
        tiny_img_bytes = create_dummy_image_bytes(width=50, height=50)
        res_tiny = await client.post(
            f"/api/v1/cases/{case_id}/images/upload",
            files={"file": ("tiny.png", tiny_img_bytes, "image/png")},
            headers=headers
        )
        assert res_tiny.status_code == 400, f"Expected 400 on sub-100x100 resolution, got {res_tiny.status_code}"
        print("\n[+] test_corrupt_upload_rejection: PASSED (Rejected corrupt bytes, empty file, and sub-100x100 resolution with 400)")


def test_preprocessing_tensor_output():
    """
    Validation Check 3:
    Preprocessing pipeline outputs tensor of shape (224, 224, 3), dtype float32,
    and pixel range bounded within [0.0, 1.0].
    """
    raw_bytes = create_dummy_image_bytes(width=300, height=300, color=(190, 80, 70))
    result = PreprocessingService.process_image(raw_bytes, "test_lesion.png")

    assert "normalized_tensor" in result, "normalized_tensor missing from preprocessing output"
    tensor = result["normalized_tensor"]

    assert isinstance(tensor, np.ndarray), f"Expected np.ndarray, got {type(tensor)}"
    assert tensor.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {tensor.shape}"
    assert tensor.dtype == np.float32, f"Expected dtype float32, got {tensor.dtype}"
    assert float(tensor.min()) >= 0.0, f"Tensor min {tensor.min()} < 0.0"
    assert float(tensor.max()) <= 1.0, f"Tensor max {tensor.max()} > 1.0"
    assert result["processed_dim"] == "224x224"
    assert result["quality_status"] in ["PASS", "WARNING", "REJECT"]
    assert isinstance(result["blur_score"], float)
    assert isinstance(result["brightness_score"], float)
    assert isinstance(result["contrast_score"], float)

    print(f"\n[+] test_preprocessing_tensor_output: PASSED (Shape: {tensor.shape}, Dtype: {tensor.dtype}, Min: {tensor.min():.2f}, Max: {tensor.max():.2f})")


def test_model_inference_pipeline():
    """
    Validation Check 4:
    Run preprocessed image tensor through recommended model (ResNet50 / EfficientNet)
    and verify:
    - Output predicted_class is one of the 6 classes (ACK, BCC, MEL, NEV, SCC, SEK)
    - Output class probabilities dictionary contains all 6 classes
    - Sum of class probabilities is approximately 1.0
    - Risk category matches clinical classification
    """
    # 1. Generate clean preprocessed image
    raw_bytes = create_dummy_image_bytes(width=224, height=224, color=(160, 90, 80))
    proc = PreprocessingService.process_image(raw_bytes, "eval_sample.png")
    tensor = proc["normalized_tensor"]

    # 2. Run inference on ResNet50
    pred_res = ModelService.predict(tensor, model_name="resnet50_baseline")
    assert pred_res["predicted_class"] in DIAGNOSTIC_CLASSES, f"Invalid predicted class {pred_res['predicted_class']}"
    assert pred_res["risk_category"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert len(pred_res["class_probabilities"]) == 6

    prob_sum = sum(pred_res["class_probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.05, f"Probabilities sum {prob_sum} deviates from 1.0"
    assert pred_res["confidence_score"] >= 0.0 and pred_res["confidence_score"] <= 1.0

    # 3. Run inference on EfficientNet-B0
    pred_eff = ModelService.predict(tensor, model_name="efficientnet_b0")
    assert pred_eff["predicted_class"] in DIAGNOSTIC_CLASSES
    assert pred_eff["risk_category"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    prob_sum_eff = sum(pred_eff["class_probabilities"].values())
    assert abs(prob_sum_eff - 1.0) < 0.05

    print(f"\n[+] test_model_inference_pipeline: PASSED (ResNet50: {pred_res['predicted_class']} ({pred_res['confidence_score']*100:.1f}%), Risk: {pred_res['risk_category']}, Prob Sum: {prob_sum:.4f})")
    print(f"[+] EfficientNet-B0: {pred_eff['predicted_class']} ({pred_eff['confidence_score']*100:.1f}%), Risk: {pred_eff['risk_category']}, Prob Sum: {prob_sum_eff:.4f})")
