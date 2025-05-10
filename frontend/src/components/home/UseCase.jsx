import { Send } from "lucide-react";

const UseCase = () => {
  return (
    <div className="w-full p-6 rounded-xl border border-gray-300 shadow-lg mb-6">
      <h2 className="text-lg font-bold text-gray-800 mb-4">Use Case Scenario</h2>

      <textarea
        className="w-full p-4 bg-[#F9FAFB] rounded-lg border border-gray-300 text-gray-700 text-sm outline-none hover:border-gray-400"
        rows="6"
        placeholder="Describe the banking scenario, including transaction details, parties involved, and any specific Shariah considerations..."
      />

      <button
        className=" place-self-end mt-4 px-6 py-2 bg-ctaBlue text-white rounded-lg flex items-center justify-center gap-2 hover:bg-blue-700 cursor-pointer transition-colors duration-300"
      >
        <Send size={16} />
        Generate Now
      </button>
    </div>
  );
};

export default UseCase;
