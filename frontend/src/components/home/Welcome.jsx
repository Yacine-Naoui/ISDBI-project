import { BookOpen } from "lucide-react";

const Welcome = () => {
  return (
    <div className="font-segoe w-full p-6 bg-gradient-to-r from-[#EFF6FF] to-[#E0E7FF] rounded-xl shadow-lg mb-6">
      <div className="flex items-center gap-[2rem]">
        <div className="flex items-center gap-4">
          <BookOpen className="text-blue-600" size={32} />
        </div>

        {/* Right: Welcome Text */}
        <div className="">
          <h2 className="text-2xl font-bold text-gray-800">Welcome to Al Murshid</h2>
          <p className=" text-sm text-gray-700 mt-2">
            We can help you identify applicable FAS Shariah standards for your banking scenarios and generate appropriate journal entries, risk reward and more! <br />
            Please describe your case.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Welcome;
