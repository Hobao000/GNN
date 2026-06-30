const API_URL = "http://127.0.0.1:8000";

export async function fetchHiSmallDemo() {
  const response = await fetch(`${API_URL}/demo/hi-small`);

  if (!response.ok) {
    throw new Error("Không thể tải dữ liệu demo.");
  }

  return response.json();
}

export async function uploadCsv(file: File) {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Predict thất bại.");
  }

  return response.json();
}