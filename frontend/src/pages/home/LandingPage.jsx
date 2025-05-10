import Layout from "../../components/layout/Layout";  // The Layout component
import Welcome from "../../components/home/Welcome";
import UseCase from "../../components/home/UseCase";
import ChatBot from "../../components/home/ChatBot";

const LandingPage = () => {
  return (
    <Layout>
      <main className="font-sogeo flex flex-col gap-5">
        {/* <Welcome /> */}
        <ChatBot />
      </main>
    </Layout>
  );
};

export default LandingPage;
