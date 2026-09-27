export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  if (req.method === "OPTIONS") return res.status(200).end();

  res.status(200).json({
    status: "ok",
    device: "vercel-serverless",
    mediapipe: false,
    environment: "vercel-preview",
    message: "KhelDrishti Vercel edge preview. Connect dedicated Python backend for live MediaPipe WebSockets."
  });
}
