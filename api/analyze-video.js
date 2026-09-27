export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();

  res.status(501).json({
    error: "backend_required",
    message: "Live video processing with MediaPipe & OpenCV requires the Python container backend. Please deploy the FastAPI backend (e.g. on Render or Railway) and set the Backend URL in the dashboard settings, or test the available pre-rendered sample movements!"
  });
}
