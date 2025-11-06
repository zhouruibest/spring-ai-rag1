package org.dx.handbook.controller;

import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentReader;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import java.io.IOException;
import com.alibaba.cloud.ai.advisor.RetrievalRerankAdvisor;
import com.alibaba.cloud.ai.model.RerankModel;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

import java.nio.charset.StandardCharsets;
import java.util.List;

@RestController
@RequestMapping("/milvus2")
public class DocumentEmbeddingController {
    private static final Logger log = LoggerFactory.getLogger(DocumentEmbeddingController.class);

    @Value("classpath:/docs/system-qa.st")
    private Resource systemResource;

    @Value("classpath:/docs/handbook.pdf")
    private Resource springAiResource;

    @Autowired
    @Qualifier("vectorStore2")
    private VectorStore vectorStore;

    @Autowired
    private  ChatModel chatModel;

    @Autowired
    private  RerankModel rerankModel;

    // Bean构建完成后执行
    @PostConstruct
    public void printInfo() {
        System.out.println("Bean构建完成！打印信息：" + chatModel.getDefaultOptions().toString());
    }

    /**
     * 处理PDF文档的解析、分割和嵌入存储。
     * 使用 PagePdfDocumentReader 解析PDF文档并生成 Document 列表。
     * 使用 TokenTextSplitter 将文档分割成更小的部分。
     * 将分割后的文档添加到向量存储中，以便后续检索和生成。
     */
    @GetMapping("/insertDocuments")
    public String insertDocuments() throws IOException {
        // 1. parse document
        DocumentReader reader = new PagePdfDocumentReader(springAiResource);
        List<Document> documents = reader.get();
        log.info("{} documents loaded", documents.size());

        // 2. split trunks
        List<Document> splitDocuments = new TokenTextSplitter().apply(documents);
        log.info("{} documents split", splitDocuments.size());

        // 3. create embedding and store to vector store
        log.info("create embedding and save to vector store");
        vectorStore.add(splitDocuments);

        return "success";
    }

    /**
     * 根据用户输入的消息生成JSON格式的聊天响应。
     * 创建一个 SearchRequest 对象，设置返回最相关的前2个结果。
     * 从 systemResource 中读取提示模板。
     * 使用 ChatClient 构建聊天客户端，调用 RetrievalRerankAdvisor 进行检索和重排序，并生成最终的聊天响应内容。
     */
    @GetMapping(value = "/ragJsonText", produces = MediaType.APPLICATION_STREAM_JSON_VALUE + ";charset=UTF-8")
    public String ragJsonText(@RequestParam(value = "message",
            defaultValue = "今夕是何年？") String message) throws IOException {

        log.info(">>>>>>>>> message: {}", message);

        SearchRequest searchRequest = SearchRequest.builder().topK(2).build();

        String promptTemplate = systemResource.getContentAsString(StandardCharsets.UTF_8);

        return ChatClient.builder(chatModel)
                .defaultAdvisors(new RetrievalRerankAdvisor(vectorStore, rerankModel, searchRequest, promptTemplate, 0.8))
                .build()
                .prompt()
                .user(message)
                .call()
                .content();
    }


    @GetMapping(value = "/ragJsonText2", produces = MediaType.APPLICATION_STREAM_JSON_VALUE + ";charset=UTF-8")
    public String ragJsonText2(@RequestParam(value = "message",
            defaultValue = "今夕是何年？") String message) throws IOException {

        SearchRequest searchRequest = SearchRequest.builder().query(message).topK(2).build();

        String promptTemplate = systemResource.getContentAsString(StandardCharsets.UTF_8);
        log.info("使用的Prompt模板: {}", promptTemplate);
        log.info("用户输入: {}", message);

        // 检索相关文档用于日志
        List<Document> relevantDocs = vectorStore.similaritySearch(searchRequest);
        log.info("检索到的文档数量: {}", relevantDocs.size());

        StringBuilder contextBuilder = new StringBuilder();
        for (Document doc : relevantDocs) {
            // 可以添加文档相似度分数的日志
            Double score = doc.getScore();
            if (score instanceof Double) {
                log.info("文档相似度分数: {}", score);
                if (score > 0.8) {
                    contextBuilder.append(doc.getContent()).append("\n\n");
                }
            }
        }
        String context = contextBuilder.toString();
        log.info("构建的上下文内容: {}", context);

        // 构建最终提交给大模型的完整prompt
        String finalPrompt = promptTemplate.replace("{{question_answer_context}}", context);
        log.info("最终提交给大模型的完整Prompt:\n{}", finalPrompt);

        return ChatClient.builder(chatModel)
                .build()
                .prompt()
                .user("用户问题: " + message + "\n" + finalPrompt)
                .call()
                .content();
    }

    /**
     * 根据用户输入的消息生成流式聊天响应。
     * 类似于 ragJsonText 方法，但使用 stream() 方法以流的形式返回聊天响应。
     * 返回类型为 Flux<ChatResponse>，适合需要实时更新的场景。
     */
    @GetMapping(value = "/ragStream", produces = "text/event-stream;charset=UTF-8")
    public Flux<ChatResponse> ragStream(@RequestParam(value = "message",
            defaultValue = "今夕是何年？") String message) throws IOException {

        log.info(">>>>>>>>> message: {}", message);

        SearchRequest searchRequest = SearchRequest.builder().topK(2).build();

        String promptTemplate = systemResource.getContentAsString(StandardCharsets.UTF_8);

        Flux<ChatResponse> chatResponses = ChatClient.builder(chatModel)
                .defaultAdvisors(new RetrievalRerankAdvisor(vectorStore, rerankModel, searchRequest, promptTemplate, 0.))
                .build()
                .prompt()
                .user(message)
                .stream()
                .chatResponse();

        return chatResponses.doOnNext(response -> {
            try {
                String content = response.getResult().getOutput().getContent();
                log.info(content);
            } catch (Exception e) {
                log.error("Error processing chat response", e);
            }
        });
    }

}