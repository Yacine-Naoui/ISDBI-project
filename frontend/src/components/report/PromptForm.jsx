import { useState } from "react";

const PromptForm = () => {
  const [formData, setFormData] = useState({
    context: "",
    buyoutPrice: "",
    bankOwnership: "",
    accountingTreatment: "",
    drEquity: "",
    crCash: "",
    additionalInputs: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Generated:", formData);
    // handle entry generation here
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-8 px-8 py-10 bg-white rounded-lg shadow-md h-full"
    >
      {[ 
        { name: "context", label: "Context" },
        { name: "buyoutPrice", label: "Buyout Price" },
        { name: "bankOwnership", label: "Bank Ownership" },
        { name: "accountingTreatment", label: "Accounting Treatment" },
        { name: "drEquity", label: "Dr Equity" },
        { name: "crCash", label: "Cr Cash" },
        { name: "additionalInputs", label: "Additional Inputs" },
      ].map((field) => (
        <div key={field.name} className="relative w-full">
          <input
            type="text"
            name={field.name}
            value={formData[field.name]}
            onChange={handleChange}
            required
            className="peer w-full border-0 border-b-2 border-gray-300 focus:outline-none focus:border-blue-600 placeholder-transparent text-base text-gray-800 py-3"
            placeholder={field.label}
          />
          <label
            htmlFor={field.name}
            className="absolute left-0 top-3 text-gray-500 text-base transition-all peer-placeholder-shown:top-3 peer-placeholder-shown:text-base peer-placeholder-shown:text-gray-400 peer-focus:top-[-10px] peer-focus:text-sm peer-focus:text-blue-600"
          >
            {field.label}
          </label>
        </div>
      ))}

      <button
        type="submit"
        className="mt-auto mx-auto bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-full transition duration-200 text-sm"
      >
        Generate Entries
      </button>
    </form>
  );
};

export default PromptForm;
