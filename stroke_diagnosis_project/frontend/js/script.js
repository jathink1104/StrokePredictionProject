function previewImage(event) {
  const imagePreview = document.getElementById('imagePreview');
  const imageInput = event.target.files[0];
  
  if (imageInput) {
      const reader = new FileReader();
      
      reader.onload = function(e) {
          imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
      };
      
      reader.readAsDataURL(imageInput);
  } else {
      imagePreview.innerHTML = ""; // Xóa hình ảnh nếu không có ảnh được chọn
  }
}

function diagnose() {
  const imageInput = document.getElementById('imageUpload');
  const modalResult = document.getElementById('modalResult');
  
  if (imageInput.files.length === 0) {
      modalResult.innerHTML = "Please upload an image first.";
      showModal();
      return;
  }

  // Giả lập quá trình chẩn đoán
  modalResult.innerHTML = "Diagnosing...";
  showModal();
  
  // Giả sử sau vài giây có kết quả
  setTimeout(() => {
      modalResult.innerHTML = "Diagnosis Complete: High risk of stroke detected.";
  }, 2000);
}

function showModal() {
  const modal = document.getElementById('resultModal');
  modal.style.display = "flex";
}

function closeModal() {
  const modal = document.getElementById('resultModal');
  modal.style.display = "none";
}

// Đóng modal khi nhấp ngoài vùng modal-content
window.onclick = function(event) {
  const modal = document.getElementById('resultModal');
  if (event.target === modal) {
      modal.style.display = "none";
  }
}


var dropdown = document.getElementById("avatarDropdown");

var avatar = document.getElementById("avatar");


avatar.onclick = function(event) {
    event.stopPropagation(); 
    dropdown.style.display = dropdown.style.display === "block" ? "none" : "block";
}


window.onclick = function(event) {
    if (!event.target.matches('.admin-avatar') && dropdown.style.display === "block") {
        dropdown.style.display = "none";
    }
}

document.addEventListener('DOMContentLoaded', function () {
  function setupModal(modalId, openBtnId, closeBtnId) {
    const modal = document.getElementById(modalId);
    const openModalBtn = document.getElementById(openBtnId);
    const closeModalBtn = document.getElementById(closeBtnId);

    if (!modal || !openModalBtn || !closeModalBtn) {
      console.warn(`Modal setup failed: Missing elements for ${modalId}`);
      return;
    }

    // Mở modal khi nhấn nút mở
    openModalBtn.addEventListener('click', () => {
      modal.style.display = 'flex';
    });

    // Đóng modal khi nhấn nút đóng
    closeModalBtn.addEventListener('click', () => {
      modal.style.display = 'none';
    });

    // Đóng modal khi nhấn bên ngoài nội dung modal
    window.addEventListener('click', (event) => {
      if (event.target === modal) {
        modal.style.display = 'none';
      }
    });
  }

  // Gọi hàm setupModal cho các modal khác nhau
  setupModal('infoModal', 'openModal', 'closeModal');
  setupModal('infoModal-Systemsetup', 'openModal-2', 'closeModal-2');
  setupModal('infoModal-ImageStory', 'openModal-3', 'closeModal-3');
  setupModal('infoModal-StorySearch', 'openModal-4', 'closeModal-4');
  setupModal('infoModal-Storyresult','openModal-5','closeModal-5');
  setupModal('infoModal-DatasetStoke','openModal-6','closeModal-6');
});


// Hiển thị hình ảnh ngay khi người dùng chọn tệp
document.getElementById("fileInput").addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();

    reader.onload = function (e) {
      const imagePreview = document.getElementById("imagePreview");

      // Xóa nội dung cũ
      imagePreview.innerHTML = "";

      // Tạo phần tử <img> để hiển thị ảnh
      const img = document.createElement("img");
      img.src = e.target.result;
      img.alt = "Selected Image";
      img.style.maxWidth = "100%";
      img.style.maxHeight = "300px";
      img.style.borderRadius = "10px";

      imagePreview.appendChild(img);
    };

    reader.readAsDataURL(file);
  }
});

// Hiển thị modal
function showModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
      modal.style.display = "flex"; // Hiển thị modal
  }
}

// Đóng modal
function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
      modal.style.display = "none"; // Ẩn modal
  }
}

// Đóng modal khi nhấn vào nút Close
document.getElementById("closeModal-result").addEventListener("click", function () {
  closeModal("resultModal");
});

// // Đóng modal khi nhấn bên ngoài nội dung modal
// window.addEventListener("click", function (event) {
//   const modal = document.getElementById("resultModal");
//   if (event.target === modal) {
//       closeModal("resultModal");
//   }
// });

// // Xử lý form upload
// document.getElementById("uploadForm").addEventListener("submit", async function (e) {
//   e.preventDefault();

//   const formData = new FormData();
//   const fileInput = document.getElementById("fileInput");
//   const file = fileInput.files[0];
//   formData.append("file", file);

//   try {
//       const response = await fetch("/predict", {
//           method: "POST",
//           body: formData,
//       });

//       if (!response.ok) {
//           alert("Error: Unable to process the image");
//           return;
//       }

//       const result = await response.json();
//       if (result.error) {
//           alert("Error: " + result.error);
//           return;
//       }

//       // Hiển thị kết quả trong modal
//       document.getElementById("prediction").innerText = result.prediction;
//       document.getElementById("confidence").innerText = result.confidence;

//       // Hiển thị ảnh đã tải lên trong modal
//       document.getElementById("uploadedImage").src = "/" + result.image_path;

//       // Mở modal
//       showModal("resultModal");
//   } catch (error) {
//       alert("Error: " + error.message);
//   }
// });




// document.getElementById("uploadForm").addEventListener("submit", function (e) {
//   e.preventDefault();

//   const formData = new FormData();
//   const fileInput = document.getElementById("fileInput");
//   formData.append("file", fileInput.files[0]);

//   fetch("/predict", {
//       method: "POST",
//       body: formData,
//   })
//       .then((response) => response.json())
//       .then((data) => {
//           if (data.error) {
//               alert(data.error);
//           } else {
//               // Hiển thị kết quả dự đoán
//               document.getElementById("prediction").innerText = data.prediction;
//               document.getElementById("confidence").innerText = data.confidence;

//               // Hiển thị ảnh gốc
//               const uploadedImage = document.getElementById("uploadedImage");
//               uploadedImage.src = data.image_path;
//               uploadedImage.alt = "Uploaded Image";

//               // Hiển thị Grad-CAM Heatmap
//               const heatmapImage = document.getElementById("heatmapImage");
//               heatmapImage.src = data.gradcam_path;
//               heatmapImage.alt = "Grad-CAM Heatmap";

//               // Hiển thị modal
//               document.getElementById("resultModal").style.display = "block";
//           }
//       })
//       .catch((error) => console.error("Error:", error));
// });

// // Đóng modal
// document.getElementById("closeModal-result").addEventListener("click", function () {
//   document.getElementById("resultModal").style.display = "none";
// });




const uploadForm = document.getElementById('uploadForm');
const resultModal = document.getElementById('resultModal');
const predictionSpan = document.getElementById('prediction');
const confidenceSpan = document.getElementById('confidence');
const uploadedImage = document.getElementById('uploadedImage');
const gradCamImage = document.getElementById('gradCamImage');
const overlayImage = document.getElementById('overlayImage');
const overlayTitle = document.getElementById('overlayTitle');
const closeModalResult = document.getElementById('closeModal-result');

uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(uploadForm);
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (response.ok) {
        predictionSpan.textContent = data.prediction;
        confidenceSpan.textContent = data.confidence;
        uploadedImage.src = data.image_path;
        gradCamImage.src = data.grad_cam_path;

        // Hiển thị ảnh overlay nếu có (nếu không phải Normal)
        if (data.overlay_path) {
            overlayImage.src = data.overlay_path;
            overlayImage.style.display = 'block';
            overlayTitle.style.display = 'block';
        } else {
            overlayImage.style.display = 'none';
            overlayTitle.style.display = 'none';
        }

        resultModal.style.display = 'block';
    } else {
        alert('Error: Unable to process the image.');
    }
});

closeModalResult.addEventListener('click', () => {
    resultModal.style.display = 'none';
});
