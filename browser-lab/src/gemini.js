const GEMINI_ORIGIN = "https://generativelanguage.googleapis.com";

function endpoint(model) {
  return `${GEMINI_ORIGIN}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
}

async function request(model, body, apiKey, signal) {
  if (!apiKey) throw new Error("Enter a Gemini API key for this tab.");
  const response = await fetch(endpoint(model), {
    method: "POST",
    mode: "cors",
    cache: "no-store",
    credentials: "omit",
    referrerPolicy: "no-referrer",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    throw new Error(`Gemini request failed with status ${response.status}.`);
  }
  return response.json();
}

function responseText(data) {
  return (data.candidates?.[0]?.content?.parts ?? [])
    .map((part) => part.text ?? "")
    .join("")
    .trim();
}

export async function translateWithGemini(prompt, apiKey, signal) {
  const data = await request(
    "gemini-2.5-flash",
    {
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        responseMimeType: "application/json",
        temperature: 0.1,
      },
    },
    apiKey,
    signal,
  );
  return { text: responseText(data), usage: data.usageMetadata ?? {} };
}

export async function synthesizeWithGemini(text, apiKey, voice, signal) {
  const data = await request(
    "gemini-2.5-flash-preview-tts",
    {
      contents: [{ role: "user", parts: [{ text }] }],
      generationConfig: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } },
        },
      },
    },
    apiKey,
    signal,
  );
  const part = (data.candidates?.[0]?.content?.parts ?? []).find((item) => item.inlineData?.data);
  if (!part) throw new Error("Gemini returned no audio.");
  return {
    pcm: base64ToBytes(part.inlineData.data),
    mimeType: part.inlineData.mimeType ?? "audio/L16;codec=pcm;rate=24000",
    usage: data.usageMetadata ?? {},
  };
}

function base64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}
