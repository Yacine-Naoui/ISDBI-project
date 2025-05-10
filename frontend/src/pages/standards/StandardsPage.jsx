import Layout from "../../components/layout/Layout";
import FASCard from "../../components/standards/FASCard";

const StandardsPage = () => {
  return (
    <Layout>
      <main>
        <div className="bg-white border border-gray-300 shadow-md rounded-md p-5 flex flex-col gap-4 h-full">
            <h2 className="text-lg font-semibold text-gray-800">FAS Standards</h2>
            
            <div className="flex flex-col gap-3 overflow-y-auto pr-1">
                {/* Repeat or map over actual FAS data here */}
                <FASCard />
                <FASCard />
                <FASCard />
                <FASCard />
                <FASCard />
                <FASCard />
            </div>
        </div>
      </main>
    </Layout>
    
  );
};

export default StandardsPage;
