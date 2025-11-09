package org.dx.handbook.config;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.ListDatabasesResponse;
import io.milvus.param.ConnectParam;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.milvus.MilvusVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class VectorStoreConfig {

    private static final Logger logger = LoggerFactory.getLogger(VectorStoreConfig.class);

    @Value("${milvus.host}")
    private String host;
    @Value("${milvus.port}")
    private Integer port;
    @Value("${milvus.database}")
    private String database;
    @Value("${milvus.collection}")
    private String collection;


    /**
     * 定义一个名为 milvusServiceClient 的Bean，用于创建并返回一个 MilvusServiceClient 实例。
     */
    @Bean
    public MilvusServiceClient milvusServiceClient() {
        try {
            // 创建 MilvusServiceClient 实例
            MilvusServiceClient client = new MilvusServiceClient(ConnectParam.newBuilder().withHost(host).withPort(port).build());

            // 测试连接状态 - 尝试列出所有数据库
            R<ListDatabasesResponse> response = client.listDatabases();

            if (response.getStatus() == R.Status.Success.getCode()) {
                logger.info("Milvus 连接成功! 服务器上的数据库数量: {}", response.getData().getDbNamesCount());
            } else {
                logger.error("Milvus 连接失败! 错误信息: {}", response.getMessage());
                throw new RuntimeException("Failed to connect to Milvus server: " + response.getMessage());
            }

            return client;
        } catch (Exception e) {
            logger.error("创建 MilvusServiceClient 失败! 主机: {}, 端口: {}", host, port, e);
            throw new RuntimeException("Failed to create MilvusServiceClient", e);
        }
    }

    /**
     * 定义一个名为 vectorStore2 的Bean，用于创建并返回一个 VectorStore 实例。
     */
    @Bean(name = "vectorStore2")
    public VectorStore vectorStore(MilvusServiceClient milvusClient, EmbeddingModel embeddingModel) {
        return MilvusVectorStore.builder(milvusClient, embeddingModel) //  嵌入模型实例，用于将文本转换为向量表示
                .collectionName(collection)
                .databaseName(database)
                .embeddingDimension(1536) //  向量嵌入维度，设置为1536，这通常与使用的嵌入模型（如OpenAI的text-embedding-ada-002）匹配
                .indexType(IndexType.IVF_FLAT) // 设置为IVF_FLAT（倒排文件Flat索引），是一种常用的近似最近邻搜索索引
                .metricType(MetricType.COSINE)
                .batchingStrategy(new TokenCountBatchingStrategy())
                .initializeSchema(false).build();
    }
}
