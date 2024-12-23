package main

import (
	"context"
	_ "embed"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/redisvector"
)

func main() {
	redisURL := "redis://127.0.0.1:6379"
	index := "test_redis_vectorstore"

	llm, e := getEmbedding("gemma:7b", "http://127.0.0.1:11434")
	ctx := context.Background()

	store, err := redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
	)
	if err != nil {
		log.Fatalln(err)
	}

	data := []schema.Document{
		{PageContent: "Tokyo", Metadata: map[string]any{"population": 9.7, "area": 622}},
		{PageContent: "Kyoto", Metadata: map[string]any{"population": 1.46, "area": 828}},
		{PageContent: "Hiroshima", Metadata: map[string]any{"population": 1.2, "area": 905}},
		{PageContent: "Kazuno", Metadata: map[string]any{"population": 0.04, "area": 707}},
		{PageContent: "Nagoya", Metadata: map[string]any{"population": 2.3, "area": 326}},
		{PageContent: "Toyota", Metadata: map[string]any{"population": 0.42, "area": 918}},
		{PageContent: "Fukuoka", Metadata: map[string]any{"population": 1.59, "area": 341}},
		{PageContent: "Paris", Metadata: map[string]any{"population": 11, "area": 105}},
		{PageContent: "London", Metadata: map[string]any{"population": 9.5, "area": 1572}},
		{PageContent: "Santiago", Metadata: map[string]any{"population": 6.9, "area": 641}},
		{PageContent: "Buenos Aires", Metadata: map[string]any{"population": 15.5, "area": 203}},
		{PageContent: "Rio de Janeiro", Metadata: map[string]any{"population": 13.7, "area": 1200}},
		{PageContent: "Sao Paulo", Metadata: map[string]any{"population": 22.6, "area": 1523}},
	}

	_, err = store.AddDocuments(ctx, data)
	if err != nil {
		log.Fatalf("Add Documents to store failed: %v", err)
	}

	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{
			PageContent: "What is the Github?",
		},
	})
	if err != nil {
		log.Fatalf("Add Documents to store failed: %v", err)
	}

	// Add documents to the Redis vector store.
	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{
			PageContent: "A city in texas",
			Metadata: map[string]any{
				"area": 3251,
			},
		},
		{
			PageContent: "A country in Asia",
			Metadata: map[string]any{
				"area": 2342,
			},
		},
		{
			PageContent: "A country in South America",
			Metadata: map[string]any{
				"area": 432,
			},
		},
		{
			PageContent: "An island nation in the Pacific Ocean",
			Metadata: map[string]any{
				"area": 6531,
			},
		},
		{
			PageContent: "A mountainous country in Europe",
			Metadata: map[string]any{
				"area": 1211,
			},
		},
		{
			PageContent: "A lost city in the Amazon",
			Metadata: map[string]any{
				"area": 1223,
			},
		},
		{
			PageContent: "A city in England",
			Metadata: map[string]any{
				"area": 4324,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	docs, err := store.SimilaritySearch(ctx, "Tokyo", 2,
		vectorstores.WithScoreThreshold(0.5),
	)
	if err != nil {
		log.Fatalf("Similarity Search for Tokyo failed: %v\n", err)
	}
	fmt.Println("Similarity Search for Tokyo, score threshold 0.5")
	fmt.Println(docs)
	fmt.Println("\n")

	docs, err = store.SimilaritySearch(ctx, "What's the github", 2,
		vectorstores.WithScoreThreshold(0.5),
	)
	if err != nil {
		log.Fatalf("Similarity Search for github failed: %v\n", err)
	}
	fmt.Println("Similarity Search for Github, score threshold 0.8")
	fmt.Println(docs)
	fmt.Println("\n")

	// Search for similar documents.
	docs, err = store.SimilaritySearch(ctx, "england", 1)
	if err != nil {
		log.Fatalf("Similarity Search for England failed: %v\n", err)
	}
	fmt.Println("Similarity Search for England")
	fmt.Println(docs)
	fmt.Println("\n")

	// Search for similar documents using score threshold.
	docs, err = store.SimilaritySearch(ctx, "american places", 10, vectorstores.WithScoreThreshold(0.80))
	if err != nil {
		log.Fatalf("Similarity Search for American places failed: %v\n", err)
	}
	fmt.Println("Similarity Search for American places with score threshold 0.80")
	fmt.Println(docs)
	fmt.Println("\n")
	/*
		// Search for similar documents using score threshold and metadata filter.
		filter := map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"key": "area",
					"range": map[string]interface{}{
						"lte": 3000,
					},
				},
			},
		}*/

	docs, err = store.SimilaritySearch(ctx, "only cities in south america",
		10,
		vectorstores.WithScoreThreshold(0.80),
		// vectorstores.WithFilters(filter))
	)
	if err != nil {
		log.Fatalf("Similarity Search for only cities in south america failed: %v\n", err)
	}
	fmt.Println("Similarity Search for only cities in south america with score threshold 0.80 and metadata filter")
	fmt.Println(docs)
	fmt.Println("\n")

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithScoreThreshold(0.8)),
		),
		"What colors is each piece of furniture next to the desk?",
	)
	if err != nil {
		log.Fatalf("Chains run failed: %v\n", err)
	}

	fmt.Println("Result of chains.Run")
	fmt.Println(result)
}

func getEmbedding(model string, connectionStr ...string) (llms.Model, *embeddings.EmbedderImpl) {
	opts := []ollama.Option{ollama.WithModel(model)}
	if len(connectionStr) > 0 {
		opts = append(opts, ollama.WithServerURL(connectionStr[0]))
	}
	llm, err := ollama.New(opts...)
	if err != nil {
		log.Fatal(err)
	}

	e, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatal(err)
	}
	return llms.Model(llm), e
}
