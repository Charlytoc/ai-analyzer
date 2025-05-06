import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

export const Markdowner = ({ markdown }: { markdown: string }) => {
  return (
    <div className="markdown-container">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node, ...props }) => {
            return <a {...props} target="_blank" rel="noopener noreferrer" />;
          },
          code: ({ node, ...props }) => {
            const isAnalysis = props.className?.includes("analysis");
            if (isAnalysis) {
              return (
                <blockquote className="bg-gray-100 p-2 rounded-md">
                  Análisis desde la palabra:{" "}
                  {props.children?.toString().split("-")[0]} hasta la palabra:{" "}
                  {props.children?.toString().split("-")[1]}
                </blockquote>
              );
            }

            return (
              <code {...props} className={`bg-gray-100 p-2 rounded-md`}>
                {props.children}
              </code>
            );
          },
        }}
      >
        {markdown}
      </Markdown>
    </div>
  );
};
