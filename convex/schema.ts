import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  predictions: defineTable({
    imageUrl: v.string(), // URL or base64
    riskScore: v.float64(), // 0 to 1
    result: v.string(), // "Low Risk", "High Risk"
    timestamp: v.number(),
    patientId: v.optional(v.string()),
  }),
});
