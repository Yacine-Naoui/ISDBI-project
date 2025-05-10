import { BookOpen, ArrowRight } from "lucide-react";

const FASCard = ({
  fasNumber = "28",
  purpose = "Murabaha and other deferred payment sales",
  section = "4.3",
  description = "Recognition & Measurement",
  arrowLink = "#",
}) => {
  return (
    <div className="flex justify-between items-center bg-gray-50 hover:bg-gray-100 transition-colors duration-200 p-4 rounded-md border border-gray-200 shadow-sm hover:shadow-md">
      {/* Left - FAS Info */}
      <div className="flex items-center gap-4">
        <BookOpen className="text-blue-600" size={24} />
        <div>
          <h4 className="font-semibold text-gray-800">{`FAS ${fasNumber} - Purpose: ${purpose}`}</h4>
          <p className="text-xs text-gray-500">{`Section ${section}: ${description}`}</p>
        </div>
      </div>

      {/* Right - Arrow */}
      <div className="flex items-center">
        <a href={arrowLink}>
          <ArrowRight
            className="text-gray-500 hover:text-blue-600 transition-colors duration-200 cursor-pointer"
            size={20}
          />
        </a>
      </div>
    </div>
  );
};

export default FASCard;
