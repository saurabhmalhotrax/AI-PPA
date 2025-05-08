"use client";

import { useEffect, useState } from "react";

interface InvoiceRecord {
  id: string;
  name: string;
  metadata: {
    vendor?: string;
    invoiceNumber?: string;
    date?: string;
    totalAmount?: string;
    [key: string]: any;
  };
  createdAt: string;
}

export default function HistoryPage() {
  const [records, setRecords] = useState<InvoiceRecord[]>([]);

  useEffect(() => {
    fetch('/api/invoices/history')
      .then((res) => res.json())
      .then((data) => setRecords(data))
      .catch((err) => console.error('Fetch history error', err));
  }, []);

  return (
    <main className="container mx-auto p-4">
      <h2 className="text-2xl font-semibold mb-4">Invoice History</h2>
      <table className="w-full table-auto border">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-4 py-2 border">ID</th>
            <th className="px-4 py-2 border">Name</th>
            <th className="px-4 py-2 border">Vendor</th>
            <th className="px-4 py-2 border">Invoice #</th>
            <th className="px-4 py-2 border">Date</th>
            <th className="px-4 py-2 border">Total</th>
          </tr>
        </thead>
        <tbody>
          {records.map((rec) => (
            <tr key={rec.id}>
              <td className="px-4 py-2 border">{rec.id}</td>
              <td className="px-4 py-2 border">{rec.name}</td>
              <td className="px-4 py-2 border">{rec.metadata.vendor || '-'}</td>
              <td className="px-4 py-2 border">{rec.metadata.invoiceNumber || '-'}</td>
              <td className="px-4 py-2 border">{rec.metadata.date || rec.createdAt}</td>
              <td className="px-4 py-2 border">{rec.metadata.totalAmount || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
} 