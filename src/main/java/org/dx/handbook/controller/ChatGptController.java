package org.dx.handbook.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
public class ChatGptController {
    private final ChatClient chatClient;
    //初始化ChatClient
    public ChatGptController(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    /**
     * 同步API
     */
    @RequestMapping("/ai/chat")
    public String chat(@RequestParam("message") String message)
    {
        return this.chatClient.prompt().user(message).call().content();
    }

    /**
     * 流式API
     */
    @RequestMapping(value = "/ai/stream",produces = "text/html;charset=utf-8")
    public Flux<String> fluxChat(@RequestParam("message")String message) {
        return chatClient.prompt().user(message).stream().content();
    }
}
