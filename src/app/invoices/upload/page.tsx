import { useState } from "react";

export default function UploadInvoicePage() {
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<any>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
  };

  const handleUpload = () => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = reader.result as string;
      try {
        const res = await fetch('/api/invoices/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: file.name, type: file.type, data: dataUrl }),
        });
        const json = await res.json();
        setMetadata(json.metadata);
      } catch (err) {
        console.error('Upload error', err);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <main className="container mx-auto p-4">
      <h2 className="text-2xl font-semibold mb-4">Upload Invoice</h2>
      <input type="file" accept="application/pdf,image/*" onChange={handleChange} />
      {file && <p className="mt-2">Selected: {file.name}</p>}
      <button
        onClick={handleUpload}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
        disabled={!file}
      >
        Upload
      </button>
      {metadata && (
        <pre className="mt-4 bg-gray-100 p-4 rounded">
          {JSON.stringify(metadata, null, 2)}
        </pre>
      )}
    </main>
  );
} 