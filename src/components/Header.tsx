import Link from "next/link";

export default function Header() {
  return (
    <header className="bg-white shadow">
      <nav className="container mx-auto px-4 py-4 flex justify-between">
        <div className="text-xl font-bold">Invoice Auditor</div>
        <div className="space-x-4 text-gray-700">
          <Link href="/">Home</Link>
          <Link href="/invoices/upload">Upload</Link>
          <Link href="/invoices/history">History</Link>
        </div>
      </nav>
    </header>
  );
} 