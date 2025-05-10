import { useState } from "react";
import { Upload, Download, ArrowLeft, Info } from "lucide-react";

const FasEnhancerSection = () => {
  const [pdfUploaded, setPdfUploaded] = useState(false);

  const handleUpload = (e) => {
    if (e.target.files.length > 0) {
      setPdfUploaded(true);
    }
  };

  return (
    <div className="w-full h-full flex flex-col items-center justify-center py-10 gap-6">
      <h2 className="text-2xl font-bold text-[#A15626] uppercase">FAS Enhancer</h2>
      <p className="max-w-md text-center text-sm text-gray-700">
        FAS enhancement refers to updates or improvements made to the Financial Accounting Standards to ensure greater transparency, accuracy, and relevance in financial reporting
      </p>

      {!pdfUploaded ? (
        <label className="bg-[#11332B] hover:bg-[#0f2b25] transition text-white font-semibold px-6 py-3 rounded-md cursor-pointer">
          <input type="file" accept="application/pdf" onChange={handleUpload} className="hidden" />
          <Upload className="inline-block mr-2" size={18} />
          Upload your PDF file
        </label>
      ) : (
        <div className="flex items-center gap-3">
          <button className="rounded-full p-2 bg-gray-800 text-white">
            <ArrowLeft size={16} />
          </button>
          <button className="bg-[#11332B] hover:bg-[#0f2b25] transition text-white font-semibold px-6 py-3 rounded-md flex items-center gap-2">
            <Download size={18} />
            Download your PDF file now
          </button>
          <div className="flex flex-col gap-1">
            <button className="rounded-full p-1.5 bg-[#A15626] text-white">
              A
            </button>
            <button className="rounded-full p-1.5 bg-[#A15626] text-white">
              I
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FasEnhancerSection;
