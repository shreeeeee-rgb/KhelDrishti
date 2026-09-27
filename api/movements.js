export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  if (req.method === "OPTIONS") return res.status(200).end();

  res.status(200).json({
    movements: {
      vertical_jump: {
        label: "Vertical Jump (CMJ)",
        benchmarks: {
          countermovement_knee: "90°–110°",
          landing_knee_flexion: "> 60° preferred, < 30° high risk",
          valgus: "< 5° safe, > 10° ACL caution"
        }
      },
      volleyball_spike: {
        label: "Volleyball Spike",
        benchmarks: {
          cocking_elbow: "> 90°",
          chain: "hips → torso → shoulder → elbow → wrist",
          landing: "quiet, symmetric, knees over toes"
        }
      },
      running_gait: {
        label: "Running Gait",
        benchmarks: {
          cadence: "170–180 steps/min",
          trunk_lean: "5°–10° forward",
          footstrike: "under the hips (no overstride)"
        }
      },
      squat: {
        label: "Squat",
        benchmarks: {
          depth: "parallel or below if mobility allows",
          spine: "no rapid pelvic tuck (butt wink)",
          knees: "track over toes, valgus < 5°"
        }
      }
    }
  });
}
