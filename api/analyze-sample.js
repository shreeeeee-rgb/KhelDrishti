import fs from "fs";
import path from "path";

export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();

  let sampleId = "vertical_jump_cmj";
  let movement = "vertical_jump";
  let language = "en";

  if (req.body) {
    if (typeof req.body === "string") {
      try {
        const parsed = JSON.parse(req.body);
        sampleId = parsed.sample_id || sampleId;
        movement = parsed.movement || movement;
        language = parsed.language || language;
      } catch (e) {
        // multipart or urlencoded
        const matchSample = req.body.match(/name="sample_id"\r\n\r\n([^\r\n]+)/);
        if (matchSample) sampleId = matchSample[1].trim();
        const matchMov = req.body.match(/name="movement"\r\n\r\n([^\r\n]+)/);
        if (matchMov) movement = matchMov[1].trim();
        const matchLang = req.body.match(/name="language"\r\n\r\n([^\r\n]+)/);
        if (matchLang) language = matchLang[1].trim();
      }
    } else if (typeof req.body === "object") {
      sampleId = req.body.sample_id || sampleId;
      movement = req.body.movement || movement;
      language = req.body.language || language;
    }
  }

  try {
    const jsonPath = path.join(process.cwd(), "sample_data", "sample_analyses.json");
    if (fs.existsSync(jsonPath)) {
      const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
      if (data[sampleId]) {
        const report = { ...data[sampleId] };
        report.source = `${sampleId}.mp4`;
        return res.status(200).json(report);
      }
      // fallback to first
      const firstKey = Object.keys(data)[0];
      if (firstKey) {
        const report = { ...data[firstKey] };
        report.source = `${firstKey}.mp4`;
        return res.status(200).json(report);
      }
    }
  } catch (err) {
    console.error("Error reading sample analysis", err);
  }

  res.status(404).json({ error: "Sample analysis not found" });
}
