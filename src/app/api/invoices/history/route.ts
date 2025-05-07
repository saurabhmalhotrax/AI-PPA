import { NextResponse } from "next/server";
import { getDocuments } from "@/lib/firebase/firebaseUtils";

export const runtime = "edge";

export async function GET() {
  try {
    const invoices = await getDocuments("invoices");
    return NextResponse.json(invoices);
  } catch (error) {
    console.error("History route error", error);
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
} 