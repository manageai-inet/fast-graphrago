package utils

import "strings"
import "time"

const DefaultDomain = "enterprise"

var DefaultEntityTypes = []string{
	"person",        // People, individuals, named persons
	"organization",  // Companies, institutions, groups, teams
	"location",      // Places, cities, countries, geographic locations
	"event",         // Occurrences, incidents, meetings, conferences
	"date",          // Time periods, dates, temporal references
	"product",       // Items, goods, services, offerings
	"technology",    // Software, tools, platforms, systems
	"concept",       // Abstract ideas, theories, principles
	"document",      // Files, reports, papers, contracts
	"project",       // Initiatives, programs, undertakings
	"role",          // Positions, titles, job functions
	"process",       // Procedures, workflows, methodologies
	"regulation",    // Rules, policies, laws, standards
	"communication", // Messages, emails, discussions, interactions
	"goal",          // Objectives, aims, targets, purposes
	"metric",        // Measurements, KPIs, indicators, statistics
	"challenge",     // Problems, obstacles, difficulties, issues
	"solution",      // Answers, fixes, resolutions, approaches
	"resource",      // Assets, materials, tools, funding
	"outcome",       // Results, consequences, achievements, impacts
	// for, components within document, they will use for mapping to asset with the same type. (these kind of asset will not be indexed.)
	"table", // Tables
	"image", // Images
	"form",  // Forms
}

const DefaultChunkSize = 500
const DefaultChunkOverlap = 50

const DefaultMaxRetryAttempt = 1
const DefaultMaxConcurrent = 5
const DefaultBatchSize = 5
const DefaultEmbedBatchSize = 32
const DefaultEmbedRetryAttempts = 3
const DefaultEmbedRetryDelay = 500 * time.Millisecond
const DefaultDedupBatchSize = 5
const DefaultMaxWalks = 100
const DefaultDampingFactor = 0.5
const DefaultTopK = 50
const DefaultThreshold = float32(0.3)

var DefaultQueryWeights = [3]float32{0.35, 0.15, 0.5}

func NormalizeName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}
