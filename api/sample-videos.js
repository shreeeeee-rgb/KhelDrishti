export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  if (req.method === "OPTIONS") return res.status(200).end();

  res.status(200).json({
    samples: [
      {
        id: "vertical_jump_cmj",
        name: "Vertical Jump Cmj",
        url: "/sample-data/vertical_jump_cmj.mp4"
      },
      {
        id: "running_gait",
        name: "Running Gait",
        url: "/sample-data/running_gait.mp4"
      },
      {
        id: "squat_atg",
        name: "Squat Atg",
        url: "/sample-data/squat_atg.mp4"
      },
      {
        id: "volleyball_spike",
        name: "Volleyball Spike",
        url: "/sample-data/volleyball_spike.mp4"
      }
    ]
  });
}
