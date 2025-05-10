import { useState } from "react";
import { Send } from "lucide-react";

const ChatBot = () => {
  const [messages, setMessages] = useState([]); // { sender: "user" | "bot", text: string }
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    // Add user message
    const newMessages = [...messages, { sender: "user", text: trimmed }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: trimmed }),
      });

      const data = await res.json();

      setMessages((prev) => [...prev, { sender: "bot", text: data.response }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Sorry, an error occurred." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full p-6 rounded-xl border border-gray-300 shadow-lg mb-6  flex flex-col gap-4">
      <h2 className="text-lg font-bold text-gray-800">Use Case Scenario</h2>

      <div className=" custom-scrollbar flex-1 max-h-[440px] min-h-[440px] overflow-y-auto bg-[#F9FAFB] p-4 rounded-lg border border-gray-300 text-sm text-gray-700 space-y-2">
        {messages.length === 0 ? (
          <p className="text-gray-400 italic">
            Start the conversation by describing the scenario...
          </p>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`p-3 rounded-lg shadow-sm ${
                msg.sender === "user"
                  ? "bg-blue-100 text-right ml-auto max-w-[80%]"
                  : "bg-white text-left mr-auto max-w-[80%]"
              }`}
            >
              {msg.text}
            </div>
          ))
        )}

        {loading && (
          <div className="p-3 rounded-lg shadow-sm bg-white text-left mr-auto max-w-[80%]">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type your message..."
          className="flex-1 p-3 bg-[#F9FAFB] rounded-lg border border-gray-300 text-gray-700 text-sm outline-none hover:border-gray-400"
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading}
          className="px-4 py-2 bg-ctaBlue text-white rounded-lg flex items-center justify-center gap-2 hover:bg-blue-700 transition-colors duration-300 disabled:opacity-50"
        >
          <Send size={16} />
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatBot;
