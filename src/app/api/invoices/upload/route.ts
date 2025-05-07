import { NextRequest, NextResponse } from "next/server";
import { addDocument } from "@/lib/firebase/firebaseUtils";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  try {
    const { name, type, data } = (await req.json()) as {
      name: string;
      type: string;
      data: string;
    };
    const openaiKey = process.env.OPENAI_API_KEY;
    if (!openaiKey) {
      console.error("Missing OpenAI API key");
      return NextResponse.json({ error: "OpenAI API key not configured" }, { status: 500 });
    }

    // Prepare chat messages for vision extraction
    const systemPrompt =
      "You are an invoice auditor. Extract the following fields from the invoice image and return as a JSON object: vendor, invoiceNumber, date, totalAmount.";
    const userContent = [
      { type: "text", text: "Extract invoice metadata from this image." },
      { type: "image_url", image_url: { url: data, detail: "low" } }
    ];

    const payload = {
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent }
      ],
      temperature: 0
    };

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openaiKey}`
      },
      method: "POST",
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("OpenAI API error", errorText);
      return NextResponse.json({ error: errorText }, { status: 500 });
    }

    const result = await response.json();
    const raw = result.choices?.[0]?.message?.content ?? "";
    let metadata: any;

    try {
      metadata = JSON.parse(raw);
    } catch {
      metadata = { raw };
    }

    // Persist to Firestore
    const docRef = await addDocument("invoices", {
      name,
      type,
      metadata,
      createdAt: new Date().toISOString()
    });

    return NextResponse.json({ metadata, id: docRef.id });
  } catch (err) {
    console.error("Upload route error", err);
    return NextResponse.json({ error: (err as Error).message }, { status: 500 });
  }
} 