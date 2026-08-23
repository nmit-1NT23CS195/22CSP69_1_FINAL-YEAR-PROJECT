const API_BASE_URL = "http://127.0.0.1:8080";

// ─────────────────────────────────────────────────────────────────
// FULL 10-QUESTION BANK  (by topic)
// ─────────────────────────────────────────────────────────────────
const QUESTION_BANK = {
  "DSA & Algorithms": [
    {
      question: "What is the time complexity of searching an element in a balanced Binary Search Tree?",
      options: ["O(n)", "O(log n)", "O(1)", "O(n log n)"],
      correctAnswer: "O(log n)",
      explanation: "In a balanced BST, each comparison eliminates half the remaining nodes, yielding O(log n) time."
    },
    {
      question: "Which data structure is ideal for implementing a BFS traversal on a graph?",
      options: ["Stack", "Priority Queue", "Queue", "Deque"],
      correctAnswer: "Queue",
      explanation: "BFS explores nodes level-by-level. A Queue (FIFO) naturally supports this by processing nodes in the order they are discovered."
    },
    {
      question: "What does the 'amortized O(1)' complexity mean for dynamic array push operations?",
      options: [
        "Every push is exactly O(1)",
        "The worst-case push is O(1)",
        "The average cost per push over a sequence is O(1)",
        "Push is O(1) only when the array is empty"
      ],
      correctAnswer: "The average cost per push over a sequence is O(1)",
      explanation: "Occasionally a resize (O(n)) occurs, but doubling the array means resizes are infrequent. Averaging across n pushes gives O(1) per operation."
    },
    {
      question: "Which sorting algorithm has O(n log n) guaranteed worst-case time complexity?",
      options: ["Quick Sort", "Merge Sort", "Heap Sort", "Both Merge Sort and Heap Sort"],
      correctAnswer: "Both Merge Sort and Heap Sort",
      explanation: "Merge Sort divides and merges in O(n log n) always. Heap Sort's heap operations are O(log n) per element, also guaranteeing O(n log n) worst-case. Quick Sort degrades to O(n²) in the worst case."
    },
    {
      question: "In Dijkstra's algorithm, what data structure is used to greedily select the minimum-distance unvisited vertex?",
      options: ["Stack", "Min-Heap / Priority Queue", "Hash Map", "Adjacency Matrix"],
      correctAnswer: "Min-Heap / Priority Queue",
      explanation: "A min-heap lets us extract the current closest node in O(log V), making the overall algorithm O((V + E) log V)."
    },
    {
      question: "Which recurrence does the Master Theorem solve as T(n) = Θ(n log n)?",
      options: [
        "T(n) = 2T(n/2) + O(n)",
        "T(n) = T(n/2) + O(1)",
        "T(n) = 4T(n/2) + O(n²)",
        "T(n) = 2T(n/2) + O(log n)"
      ],
      correctAnswer: "T(n) = 2T(n/2) + O(n)",
      explanation: "Case 2 of the Master Theorem applies when f(n) = Θ(n^log_b(a)). Here a=2, b=2, so n^log₂2 = n, matching f(n)=O(n). Solution: T(n) = Θ(n log n)."
    },
    {
      question: "A hash table has load factor α = n/m. What collision resolution ensures O(1) average lookup when α < 1?",
      options: ["Open Addressing (Linear Probing)", "Separate Chaining with balanced BSTs", "Double Hashing", "All of the above"],
      correctAnswer: "All of the above",
      explanation: "Under α < 1 with good hash functions, all these methods provide expected O(1) lookup. Chaining with BSTs ensures O(log k) per chain, but short chains keep average near O(1)."
    },
    {
      question: "What is the space complexity of DFS on a graph with V vertices and E edges?",
      options: ["O(V + E)", "O(V)", "O(E)", "O(log V)"],
      correctAnswer: "O(V)",
      explanation: "DFS uses an implicit (or explicit) call stack of depth at most V (one frame per vertex on the path). The visited array is O(V). Adjacency list is O(V+E) but that's input space."
    },
    {
      question: "Which algorithm solves the All-Pairs Shortest Path problem in O(V³)?",
      options: ["Bellman-Ford", "Dijkstra", "Floyd-Warshall", "Johnson's Algorithm"],
      correctAnswer: "Floyd-Warshall",
      explanation: "Floyd-Warshall uses dynamic programming with a 3-nested loop over all pairs (i, j) via intermediate vertex k, yielding O(V³) time and O(V²) space."
    },
    {
      question: "In a min-heap of n elements, what is the time complexity of the extract-min operation?",
      options: ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
      correctAnswer: "O(log n)",
      explanation: "Extract-min removes the root, replaces it with the last element, then heapifies down. Heapify-down runs in O(log n) because the tree height is ⌊log₂ n⌋."
    }
  ],
  "Operating Systems": [
    {
      question: "What is the primary difference between a process and a thread?",
      options: [
        "A process has its own memory space; threads share the parent process's memory",
        "Threads have their own memory space; processes share memory",
        "Processes are lighter-weight than threads",
        "There is no difference in modern OSes"
      ],
      correctAnswer: "A process has its own memory space; threads share the parent process's memory",
      explanation: "Processes are isolated — they have separate address spaces, file handles, etc. Threads within a process share the heap and globals, communicating faster but risking race conditions."
    },
    {
      question: "Which page replacement algorithm suffers from Bélády's Anomaly?",
      options: ["LRU (Least Recently Used)", "Optimal", "FIFO (First-In-First-Out)", "LFU (Least Frequently Used)"],
      correctAnswer: "FIFO (First-In-First-Out)",
      explanation: "Bélády's Anomaly is the counterintuitive result where adding more page frames can increase page faults with FIFO. LRU and Optimal do not exhibit this anomaly."
    },
    {
      question: "In the context of deadlock, which of the four Coffman conditions is eliminated by a 'resource ordering' strategy?",
      options: ["Mutual Exclusion", "Hold and Wait", "No Preemption", "Circular Wait"],
      correctAnswer: "Circular Wait",
      explanation: "Imposing a global ordering on resource types prevents circular chains of dependencies, eliminating the Circular Wait condition."
    },
    {
      question: "What does the term 'thrashing' mean in operating systems?",
      options: [
        "High CPU utilization with productive work",
        "A process repeatedly crashing due to a bug",
        "Excessive paging where a process spends more time swapping than executing",
        "An OS scheduling loop that starves low-priority processes"
      ],
      correctAnswer: "Excessive paging where a process spends more time swapping than executing",
      explanation: "Thrashing occurs when the working set of processes exceeds available physical memory, causing near-constant page faults and minimal useful CPU work."
    },
    {
      question: "Which scheduling algorithm minimizes average waiting time for a given set of processes (non-preemptive)?",
      options: ["Round Robin", "First Come First Serve (FCFS)", "Shortest Job First (SJF)", "Priority Scheduling"],
      correctAnswer: "Shortest Job First (SJF)",
      explanation: "Non-preemptive SJF provably minimizes average waiting time by always selecting the burst with the smallest remaining time next. It is optimal for this metric in the non-preemptive setting."
    },
    {
      question: "A semaphore initialized to 1 is used to protect a critical section. What type of semaphore is this?",
      options: ["Counting semaphore", "Binary semaphore (mutex)", "Spinlock semaphore", "Read-write semaphore"],
      correctAnswer: "Binary semaphore (mutex)",
      explanation: "A semaphore initialized to 1 can only take values 0 or 1, functioning as a mutual exclusion lock — a binary semaphore or mutex."
    },
    {
      question: "In a virtual memory system, what is the Translation Lookaside Buffer (TLB)?",
      options: [
        "A disk cache for swap space",
        "A small, fast associative cache that stores recent virtual-to-physical page translations",
        "A hardware register holding the base address of the page table",
        "A kernel data structure for tracking free physical frames"
      ],
      correctAnswer: "A small, fast associative cache that stores recent virtual-to-physical page translations",
      explanation: "The TLB acts as a cache for page table entries, reducing the cost of virtual address translation from multiple memory accesses to a single hardware lookup on a hit."
    },
    {
      question: "Which inter-process communication (IPC) mechanism provides the fastest communication between processes on the same machine?",
      options: ["Sockets", "Message Queues", "Pipes", "Shared Memory"],
      correctAnswer: "Shared Memory",
      explanation: "Shared memory allows processes to read/write a common region directly without kernel involvement per access, making it the fastest IPC mechanism. Other mechanisms require kernel mediation on each operation."
    },
    {
      question: "What is a 'zombie process' in Unix/Linux?",
      options: [
        "A process that consumes 100% CPU indefinitely",
        "A child process that has terminated but whose exit status has not yet been collected by the parent",
        "A background daemon process with no controlling terminal",
        "A process stuck in an uninterruptible sleep state"
      ],
      correctAnswer: "A child process that has terminated but whose exit status has not yet been collected by the parent",
      explanation: "When a child exits, its PCB remains until the parent calls wait(). Until then, the entry is a 'zombie' — dead but not yet reaped."
    },
    {
      question: "Which memory allocation strategy results in the smallest leftover fragment (external fragmentation)?",
      options: ["First Fit", "Best Fit", "Worst Fit", "Next Fit"],
      correctAnswer: "Best Fit",
      explanation: "Best Fit allocates the smallest hole that is large enough for the request, minimizing leftover space per allocation. However, it tends to create many tiny unusable fragments over time."
    }
  ],
  "Computer Networks": [
    {
      question: "Which layer of the OSI model is responsible for logical addressing and routing?",
      options: ["Data Link Layer (Layer 2)", "Network Layer (Layer 3)", "Transport Layer (Layer 4)", "Session Layer (Layer 5)"],
      correctAnswer: "Network Layer (Layer 3)",
      explanation: "The Network Layer (L3) handles logical IP addressing, packet routing, and forwarding across multiple networks. Routers operate at this layer."
    },
    {
      question: "What is the purpose of the TCP three-way handshake?",
      options: [
        "To encrypt data before transmission",
        "To establish a reliable, synchronized connection between client and server",
        "To negotiate the maximum transmission unit (MTU)",
        "To authenticate the client's identity to the server"
      ],
      correctAnswer: "To establish a reliable, synchronized connection between client and server",
      explanation: "SYN → SYN-ACK → ACK synchronizes sequence numbers and confirms both sides can send and receive, establishing state before data transfer."
    },
    {
      question: "Which congestion control algorithm in TCP reduces the congestion window to 1 MSS upon detecting a timeout?",
      options: ["Fast Retransmit", "Fast Recovery (Reno)", "Slow Start", "CUBIC"],
      correctAnswer: "Slow Start",
      explanation: "On a timeout (severe congestion signal), TCP resets cwnd to 1 MSS and enters Slow Start phase. It also halves ssthresh. Fast Recovery handles triple-dup-ACK differently."
    },
    {
      question: "What does CIDR (Classless Inter-Domain Routing) improve over classful addressing?",
      options: [
        "Encryption of routing tables",
        "Flexible subnet sizing and reduced routing table bloat",
        "Automatic IP assignment to hosts",
        "Priority-based packet scheduling"
      ],
      correctAnswer: "Flexible subnet sizing and reduced routing table bloat",
      explanation: "CIDR uses prefix-length notation (e.g., /22) allowing arbitrary block sizes instead of rigid Class A/B/C boundaries, enabling efficient IP allocation and route aggregation."
    },
    {
      question: "In HTTP/2, what technique allows the server to push resources proactively before the client requests them?",
      options: ["Long Polling", "WebSockets", "Server Push", "HTTP Pipelining"],
      correctAnswer: "Server Push",
      explanation: "HTTP/2 Server Push lets the server pre-send resources (e.g., CSS, JS) it anticipates the client will need, reducing round-trips and latency."
    },
    {
      question: "What distinguishes UDP from TCP in terms of reliability?",
      options: [
        "UDP guarantees delivery; TCP does not",
        "UDP has no error-checking; TCP uses checksums",
        "UDP provides no delivery guarantees or ordering; TCP ensures reliability, ordering, and flow control",
        "UDP is connection-oriented; TCP is connectionless"
      ],
      correctAnswer: "UDP provides no delivery guarantees or ordering; TCP ensures reliability, ordering, and flow control",
      explanation: "UDP is a 'fire-and-forget' protocol — minimal overhead, no retransmission, no ordering. It suits real-time applications (video, DNS) where speed > reliability."
    },
    {
      question: "What is the function of ARP (Address Resolution Protocol)?",
      options: [
        "Maps domain names to IP addresses",
        "Maps IP addresses to MAC addresses on the same local network segment",
        "Assigns dynamic IP addresses to hosts",
        "Routes packets across autonomous systems"
      ],
      correctAnswer: "Maps IP addresses to MAC addresses on the same local network segment",
      explanation: "ARP broadcasts a query 'Who has IP X?' on the LAN. The host with that IP replies with its MAC address, enabling the sender to build an Ethernet frame."
    },
    {
      question: "A company has the IP block 192.168.10.0/24. How many usable host addresses are available?",
      options: ["254", "256", "255", "252"],
      correctAnswer: "254",
      explanation: "A /24 block has 2⁸ = 256 addresses. Subtract the network address (x.x.x.0) and broadcast address (x.x.x.255), leaving 254 usable host addresses."
    },
    {
      question: "Which DNS record type maps a domain name to an IPv4 address?",
      options: ["AAAA", "CNAME", "A", "MX"],
      correctAnswer: "A",
      explanation: "An 'A' record (Address record) maps a hostname to a 32-bit IPv4 address. AAAA maps to IPv6, CNAME is an alias, MX designates mail servers."
    },
    {
      question: "What is the primary role of a reverse proxy?",
      options: [
        "Forwards client requests to the internet on behalf of the client (hides client identity)",
        "Sits in front of servers, forwarding requests from clients to backend servers (hides server identity, enables load balancing)",
        "Caches DNS responses to reduce lookup latency",
        "Encrypts data at the transport layer using TLS"
      ],
      correctAnswer: "Sits in front of servers, forwarding requests from clients to backend servers (hides server identity, enables load balancing)",
      explanation: "A reverse proxy (e.g., Nginx, Cloudflare) terminates client connections and forwards to one of many backend servers, enabling load balancing, SSL termination, and caching."
    }
  ],
  "Web Development & React": [
    {
      question: "In React, what is the purpose of the dependency array in the useEffect hook?",
      options: [
        "It lists props that the component accepts",
        "It controls when the effect runs — it re-runs only when the listed values change",
        "It specifies which state variables the effect is allowed to mutate",
        "It defines the cleanup function for the effect"
      ],
      correctAnswer: "It controls when the effect runs — it re-runs only when the listed values change",
      explanation: "useEffect(fn, [dep1, dep2]) runs fn after mount and after any render where dep1 or dep2 changed. An empty array [] means run-once on mount. Omitting the array means run on every render."
    },
    {
      question: "What is the Virtual DOM in React, and why does it improve performance?",
      options: [
        "A copy of the DOM stored on the server for SSR",
        "A lightweight JavaScript object tree that React uses to diff and batch DOM updates efficiently",
        "A browser-native API for off-screen rendering",
        "A React-internal cache of previously fetched API responses"
      ],
      correctAnswer: "A lightweight JavaScript object tree that React uses to diff and batch DOM updates efficiently",
      explanation: "React maintains a VDOM in memory. On state change, it generates a new VDOM, diffs it with the previous one (reconciliation), and applies only the minimal set of real DOM mutations."
    },
    {
      question: "What does CSS specificity determine?",
      options: [
        "The order in which CSS files are loaded",
        "Which CSS rule wins when multiple conflicting rules target the same element",
        "The rendering priority of elements on the page",
        "Whether a style is applied before or after JavaScript execution"
      ],
      correctAnswer: "Which CSS rule wins when multiple conflicting rules target the same element",
      explanation: "Specificity is calculated as (inline, IDs, classes/attrs, elements). Higher specificity wins. Inline > ID > class/pseudo-class/attribute > element/pseudo-element."
    },
    {
      question: "In HTTP, what is the difference between PUT and PATCH?",
      options: [
        "PUT creates a resource; PATCH updates it",
        "PUT replaces the entire resource; PATCH applies a partial update",
        "PUT is idempotent; PATCH is not",
        "They are identical except PATCH requires authentication"
      ],
      correctAnswer: "PUT replaces the entire resource; PATCH applies a partial update",
      explanation: "PUT sends a complete representation of the resource. PATCH sends only the fields to be modified. Both should ideally be idempotent, though PATCH implementations vary."
    },
    {
      question: "What is the output of: console.log(typeof null) in JavaScript?",
      options: ["'null'", "'undefined'", "'object'", "'boolean'"],
      correctAnswer: "'object'",
      explanation: "This is a well-known JavaScript bug from its first implementation. typeof null === 'object' despite null not being an object. To check for null, use value === null."
    },
    {
      question: "What does the SQL statement 'SELECT * FROM orders WHERE ROWNUM <= 10' retrieve, and which DB engine does this syntax belong to?",
      options: [
        "First 10 rows — MySQL syntax",
        "First 10 rows — Oracle syntax",
        "Random 10 rows — PostgreSQL syntax",
        "Last 10 rows — SQL Server syntax"
      ],
      correctAnswer: "First 10 rows — Oracle syntax",
      explanation: "ROWNUM is an Oracle pseudo-column. MySQL uses LIMIT 10, PostgreSQL uses LIMIT 10 or FETCH FIRST 10 ROWS ONLY, SQL Server uses TOP 10."
    },
    {
      question: "In React, what is 'prop drilling' and what is the standard solution?",
      options: [
        "Prop drilling is a performance optimization; use React.memo to solve it",
        "Prop drilling is passing props through many intermediate components; solve with Context API or state management libraries",
        "Prop drilling is a security vulnerability in JSX; solve with PropTypes validation",
        "Prop drilling refers to deeply nested hooks; solve with custom hooks"
      ],
      correctAnswer: "Prop drilling is passing props through many intermediate components; solve with Context API or state management libraries",
      explanation: "When data must pass through many layers of components that don't use it themselves, use React Context, Zustand, Redux, or Jotai to make state globally accessible."
    },
    {
      question: "What is a JavaScript closure?",
      options: [
        "A method to close a browser tab via JavaScript",
        "A function that retains access to variables from its outer (lexical) scope even after the outer function has returned",
        "An IIFE (Immediately Invoked Function Expression)",
        "A way to prevent a function from being called more than once"
      ],
      correctAnswer: "A function that retains access to variables from its outer (lexical) scope even after the outer function has returned",
      explanation: "Closures allow inner functions to 'close over' variables of their enclosing scope. This is fundamental to data encapsulation, factory functions, and hooks like useState."
    },
    {
      question: "What HTTP status code indicates 'Too Many Requests' (rate limiting)?",
      options: ["400", "403", "429", "503"],
      correctAnswer: "429",
      explanation: "429 Too Many Requests is the standard response when a client exceeds a rate limit. The response should include a Retry-After header indicating when to try again."
    },
    {
      question: "In a RESTful API, which HTTP method is used to retrieve a resource without side effects?",
      options: ["POST", "PUT", "GET", "DELETE"],
      correctAnswer: "GET",
      explanation: "GET is idempotent and safe — it retrieves data without modifying server state. POST creates, PUT replaces, DELETE removes. GET responses should be cacheable."
    }
  ]
};

// Flatten all questions and pick 10 at random, or use topic-specific set
function getRandomQuestions(topic, count = 10) {
  const pool = topic && QUESTION_BANK[topic]
    ? QUESTION_BANK[topic]
    : Object.values(QUESTION_BANK).flat();

  const shuffled = [...pool].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, shuffled.length));
}

export async function generateQuiz(topic) {
  try {
    // Uncomment when FastAPI backend endpoint is ready:
    // const url = topic
    //   ? `${API_BASE_URL}/api/mcq/questions?topic=${encodeURIComponent(topic)}&count=10`
    //   : `${API_BASE_URL}/api/mcq/questions?count=10`;
    // const response = await fetch(url);
    // if (!response.ok) throw new Error("Failed to fetch questions from backend");
    // return await response.json();

    // Simulated async delay for realistic loading
    await new Promise(resolve => setTimeout(resolve, 600));
    return getRandomQuestions(topic, 10);
  } catch (error) {
    console.error("Error in generateQuiz:", error);
    throw error;
  }
}

export async function saveQuizResult(questions, answers, score) {
  try {
    // Uncomment when FastAPI backend endpoint is ready:
    // const response = await fetch(`${API_BASE_URL}/api/mcq/submit`, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify({
    //     questions,
    //     answers,
    //     score,
    //     timestamp: new Date().toISOString(),
    //   }),
    // });
    // if (!response.ok) throw new Error("Failed to save result to backend");
    // return await response.json();

    console.log("[PlaceBuddy] MCQ result saved (local):", { score: score.toFixed(1) + '%', totalQuestions: questions.length });

    // Save to localStorage for persistence
    const resultRecord = {
      id: `mcq_${Date.now()}`,
      score,
      totalQuestions: questions.length,
      correctCount: answers.filter((a, i) => a === questions[i]?.correctAnswer).length,
      timestamp: new Date().toISOString(),
    };
    try {
      const existing = JSON.parse(localStorage.getItem('placebuddy_mcq_history') || '[]');
      const updated = [resultRecord, ...existing].slice(0, 10);
      localStorage.setItem('placebuddy_mcq_history', JSON.stringify(updated));
    } catch (_) {}

    return resultRecord;
  } catch (error) {
    console.error("Error in saveQuizResult:", error);
    throw error;
  }
}

export async function getMcqHistory() {
  try {
    const stored = localStorage.getItem('placebuddy_mcq_history');
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

export { QUESTION_BANK };
