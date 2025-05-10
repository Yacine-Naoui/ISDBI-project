import { FileText, Download } from "lucide-react";
import * as XLSX from "xlsx"; // Importing the xlsx library

const JournalEntries = () => {
  // Dummy data for the journal entries table
  const journalData = [
    { account: "Murabaha Inventory", debit: 10000, credit: 0 },
    { account: "Cash", debit: 0, credit: 10000 },
    { account: "Receivables", debit: 5000, credit: 0 },
    { account: "Payables", debit: 0, credit: 5000 },
    { account: "Deferred Revenue", debit: 0, credit: 2000 },
  ];

  const handleExport = () => {
    // Create a worksheet from the journal data
    const ws = XLSX.utils.json_to_sheet(journalData);
    
    // Create a new workbook and append the worksheet
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Journal Entries");
    
    // Export the workbook to a file
    XLSX.writeFile(wb, "journal_entries.xlsx");
  };

  return (
    <div className="border border-gray-200 rounded-lg shadow-sm p-6 bg-white">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-2">
          <FileText className="text-gray-700" size={24} />
          <h3 className="text-lg font-semibold text-gray-800">Journal Entries</h3>
        </div>
        <button
          onClick={handleExport}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-blue-700 cursor-pointer"
        >
          <Download size={18} />
          Export
        </button>
      </div>

      {/* Table */}
      <table className="min-w-full text-sm rounded-lg text-gray-700">
        <thead className="">
          <tr className="">
            <th className="py-2 px-4 text-left bg-gray-200 rounded-tl-lg">Account</th>
            <th className="py-2 px-4 text-left bg-gray-200">Debit</th>
            <th className="py-2 px-4 text-left bg-gray-200 rounded-tr-lg">Credit</th>
          </tr>
        </thead>
        <tbody>
          {journalData.map((entry, index) => (
            <tr key={index} className="border-b border-gray-200">
              <td className="py-2 px-4 bg-gray-100">{entry.account}</td>
              <td className="py-2 px-4 bg-gray-100">{entry.debit}</td>
              <td className="py-2 px-4 bg-gray-100">{entry.credit}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default JournalEntries;
