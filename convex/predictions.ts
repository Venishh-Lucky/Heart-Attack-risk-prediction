import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const savePrediction = mutation({
  args: {
    imageUrl: v.string(),
    riskScore: v.float64(),
    result: v.string(),
    timestamp: v.number(),
  },
  handler: async (ctx, args) => {
    const id = await ctx.db.insert("predictions", {
      imageUrl: args.imageUrl,
      riskScore: args.riskScore,
      result: args.result,
      timestamp: args.timestamp,
    });
    return id;
  },
});

export const getHistory = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db
      .query("predictions")
      .order("desc")
      .take(10);
  },
});
