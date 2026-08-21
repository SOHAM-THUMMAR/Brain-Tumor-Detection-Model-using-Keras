document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const filePreview = document.getElementById('file-preview-name');
  const submitBtn = document.getElementById('submit-btn');
  const uploadForm = document.getElementById('upload-form');

  if (!dropzone || !fileInput) return;

  const allowedExtensions = ['png', 'jpg', 'jpeg'];

  const validateFile = (file) => {
    if (!file) return false;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(ext)) {
      alert(`Invalid file format (.${ext}). Please select a PNG, JPG, or JPEG image.`);
      fileInput.value = '';
      if (filePreview) filePreview.textContent = '';
      if (submitBtn) submitBtn.disabled = true;
      return false;
    }
    if (filePreview) {
      filePreview.textContent = `Selected file: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    }
    if (submitBtn) submitBtn.disabled = false;
    return true;
  };

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      fileInput.files = files;
      validateFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) {
      validateFile(fileInput.files[0]);
    }
  });

  if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
      if (submitBtn) {
        submitBtn.innerHTML = 'Analyzing MRI Scan... <span class="spinner"></span>';
        submitBtn.style.opacity = '0.8';
        submitBtn.disabled = true;
      }
    });
  }
});
