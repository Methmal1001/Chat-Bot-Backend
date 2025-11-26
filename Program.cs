using Microsoft.AspNetCore.Mvc;
using Microsoft.Data.SqlClient;
using Dapper;
using Helpers;   // IMPORTANT → this loads TextSimilarity from Helpers folder
using System.Text.RegularExpressions;

// -------------------------------------------
// Build App
// -------------------------------------------
var builder = WebApplication.CreateBuilder(args);

// Add Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Allow Nuxt 3 frontend
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:3000")
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

var app = builder.Build();

app.UseCors();
app.UseSwagger();
app.UseSwaggerUI();

// Read connection string
string db = builder.Configuration.GetConnectionString("DefaultConnection");


// -----------------------------------------------------------
// POST /chat  → AI-powered TF-IDF similarity chatbot
// -----------------------------------------------------------
app.MapPost("/chat", async ([FromBody] ChatRequest req) =>
{
    if (string.IsNullOrWhiteSpace(req.Message))
        return Results.BadRequest(new { error = "Message cannot be empty" });

    string userMsg = req.Message.ToLower();

    try
    {
        using var conn = new SqlConnection(db);

        // Load all chatbot triggers + responses
        var records = await conn.QueryAsync<(string Trigger, string Response)>(
            "SELECT [Trigger], [Response] FROM ChatBotResponses"
        );

        if (!records.Any())
            return Results.Ok(new { reply = "No data found in knowledge base." });

        // Convert each trigger into tokenized documents
        var docs = records.Select(r => TextSimilarity.Tokenize(r.Trigger)).ToList();

        // Compute IDF
        var idf = TextSimilarity.InverseDocumentFrequency(docs);

        // Convert user message → TF-IDF vector
        var userTokens = TextSimilarity.Tokenize(userMsg);
        var userVector = TextSimilarity.TfIdfVector(userTokens, idf);

        double bestScore = 0;
        string bestReply = "Sorry, I couldn't understand that. Can you rephrase?";

        // Compare similarity with all trigger texts
        foreach (var r in records)
        {
            var trigTokens = TextSimilarity.Tokenize(r.Trigger);
            var trigVector = TextSimilarity.TfIdfVector(trigTokens, idf);

            double score = TextSimilarity.CosineSimilarity(userVector, trigVector);

            if (score > bestScore)
            {
                bestScore = score;
                bestReply = r.Response;
            }
        }

        // Save chat to DB
        await conn.ExecuteAsync(
            "INSERT INTO ChatMessages(UserMessage, BotMessage) VALUES(@u, @b)",
            new { u = req.Message, b = bestReply }
        );

        return Results.Ok(new { reply = bestReply });
    }
    catch (Exception ex)
    {
        return Results.Problem("Error: " + ex.Message);
    }
});


// -----------------------------------------------------------
// GET /history  → fetch stored chat logs
// -----------------------------------------------------------
app.MapGet("/history", async () =>
{
    using var conn = new SqlConnection(db);
    var messages = await conn.QueryAsync<ChatMessage>(
        "SELECT * FROM ChatMessages ORDER BY CreatedAt ASC"
    );
    return Results.Ok(messages);
});


app.Run();


// =============================================================
// Models
// =============================================================
public record ChatRequest(string Message);

public class ChatMessage
{
    public int Id { get; set; }
    public string UserMessage { get; set; }
    public string BotMessage { get; set; }
    public DateTime CreatedAt { get; set; }
}
