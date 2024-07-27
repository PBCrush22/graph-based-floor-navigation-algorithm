import math


class Vertex:
    def __init__(self, id: int):
        """
        Function description: Initialise the Vertex's id and connections as a list
        :Precondition: None
        :Postcondition: The Vertex object is initialised
        :Input:
            id: The Vertex's id
        :Return: None
        :Time complexity:
            Worst: O(1)
            Best: same as worst case
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        self.id = id
        self.connections = []


class Edge:
    def __init__(self, u: Vertex, v: Vertex, weight: int):
        """
        Function description: Initialise the Edge's source vertex, exit vertex and weight
        :Precondition: The Vertex objects u and v have been initialised
        :Postcondition: The Edge object is initialised
        :Input:
            u: The Edge's source vertex (Vertex object)
            v: The Edge's exit vertex (Vertex object)
            weight: The Edge's weight
        :Return: None
        :Time complexity:
            Worst: O(1)
            Best: same as worst case
        :Space complexity:
            Input: O(V) - the Vertex objects u and v
            Auxiliary: O(1)
        """
        self.u = u
        self.v = v
        self.w = weight


class MinHeap:
    def __init__(self):
        """
        Function description: Initialise the MinHeap's array and count
        :Precondition: None
        :Postcondition: The MinHeap's array is initialised
        :Input: None
        :Return: None
        :Time complexity: O(1)
        :Space complexity: O(1)
        """
        self.array = [None]
        self.length = 0

    def __len__(self):
        """
        Function description: Return the length of the MinHeap
        :Input: None
        :Return: The length of the MinHeap
        :Time complexity:
            Worst: O(1)
            Best: same as worst case
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        return self.length

    def add(self, key, val):
        """
        Function description: Add a new entry to the MinHeap
        :Precondition: None
        :Postcondition: The new entry is added to the MinHeap
        :Input:
            key: The key of the new entry
            val: The value of the new entry
        :Return: None
        :Time complexity:
            Worst: O(log n), which is the worst case of rise()
            Best: same as worst case
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        self.array.append([key, val])
        self.length += 1
        self.rise(self.length)

    def swap(self, a, b):
        """
        Function description: Swap two entries in the MinHeap
        :Input:
            a: The index of the first entry
            b: The index of the second entry
        :Return: None
        :Time complexity: O(1)
        :Space complexity: O(1)
        """
        self.array[a], self.array[b] = self.array[b], self.array[a]

    def rise(self, k):
        """
        Function description: Rise an entry in the MinHeap
        :Precondition: 1 <= k <= self.length
        :Postcondition: The entry is risen
        :Input:
            k: The index of the entry to rise
        :Return: None
        :Time complexity:
            Worst: O(log n), where log n is the height of the MinHeap
            Best: O(1), when no swaps is required
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        while k > 1 and self.array[k][0] < self.array[k // 2][0]:
            self.swap(k, k // 2)
            k //= 2

    def sink(self, k):
        """
        Function description: Sink an entry in the MinHeap
        :Precondition: 1 <= k <= self.length
        :Postcondition: The entry is sunk
        :Input:
            k: The index of the entry to sink
        :Return: None
        :Time complexity:
            Worst: O(log n), where log n is the height of the MinHeap
            Best: O(1), when no swaps is required
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        while 2 * k <= self.length:
            child = self.smallest_child(k)
            if self.array[k][0] <= self.array[child][0]:
                break
            self.swap(child, k)
            k = child

    def get_min(self):
        """
        Function description: Remove and return the minimum entry in the MinHeap
        :Precondition: self.length >= 1
        :Postcondition: The minimum entry is removed and returned from the MinHeap
        :Input: None
        :Return: The minimum entry
        :Time complexity:
            Worst: O(log n), which is the worst case of sink()
            Best: O(1), when there is only one entry in the MinHeap
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        self.swap(1, self.length)
        min = self.array.pop(self.length)
        self.length -= 1
        self.sink(1)
        return min

    def smallest_child(self, k):
        """
        Function description: Return the index of k's child with smallest value
        :Precondition: 1 <= k <= self.length
        :Postcondition: None
        :Input:
            k: The index of the entry
        :Return: The index of the smallest child of the entry
        :Time complexity:
            Worst: O(1)
            Best: same as worst case
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        if 2 * k == self.length or self.array[2 * k][0] < self.array[2 * k + 1][0]:
            return 2 * k
        else:
            return 2 * k + 1


class FloorGraph:
    ### For all complexity stated in this class, V is the number of vertices and E is the number of edges. ###
    def __init__(self, paths: list, keys: list):
        """
        Function description: Initialise the FloorGraph's vertex_list, edge_list, max_vertex_id and keys
        :Precondition: None
        :Postcondition: The FloorGraph object is initialised
        :Input:
            paths: A list of tuples containing u, v and w of each edge
            keys: A list of tuples containing the key's location index and time needed to get key
        :Return: None
        :Time complexity:
            Worst: O(V + E)
            Best: same as worst case
        :Space complexity:
            Input: O(V + E)
            Auxiliary: O(V + E)
        """
        self.vertex_list = []  # Store Vertex objects of the graph
        self.edge_list = []  # Store Edge objects of the graph
        self.max_vertex_id = 0
        self.keys = keys

        # Get the highest vertex id
        for i in range(len(paths)):  # O(E)
            u, v = paths[i][:2]  # Get the vertices id of every path
            if max(u, v) > self.max_vertex_id:
                self.max_vertex_id = max(u, v)

        # Populate the vertex_list with Vertex objects based on the highest vertex id
        for j in range(self.max_vertex_id + 1):  # O(V)
            self.vertex_list.append(Vertex(j))

        # Populate edge_list and Vertex connections
        for i in range(len(paths)):  # O(E)
            # Get the vertices connected by the edge and the weight of the edge
            from_vertex = self.vertex_list[paths[i][0]]
            to_vertex = self.vertex_list[paths[i][1]]
            weight = paths[i][2]

            edge = Edge(from_vertex, to_vertex, weight)  # Create edge

            self.edge_list.append(edge)  # edge_list stores Edge objects

            from_vertex.connections.append(
                edge
            )  # Add edge to from_vertex's connections

    def climb(self, start: int, exits: list):
        """
        Function description: Return one shortest route from start to one of the exits points, including the time to defeat one monster and collect the key.
        Approach description:
            1. Add a dummy node to the graph, connect all exits vertex to it - O(E)
            2. Run forward Dijkstra's - O(E log V)
            3. Run backward's Dijkstra's - O(E log V)
            4. From the two graphs above, identify the path from start to dummy node with the shortest travel + fighting monster time. - O(V)
            5. Bring everything back to normal state - O(E)

        :Precondition: The Graph has been initialised
        :Postcondition: None

        :Input:
            start: A number indicating the index of the start vertex
            exits: A list of numbers indicating the index of the exit vertices
        :Return:
            tuple containing:
                total time (required from the start from one of the exits, plus the time to fight a monster to get a key)
                path (list of vertices to go through to get to the exit)
            OR
            None (if no path exists between start vertex and any exits vertex)
        :Time complexity:
            Worst: O(E log V)
            Best: same as worst case
        :Space complexity:
            Input: O(V) - the exits list
            Auxiliary: O(E + V)
        """
        # add a dummy node to the graph, that connects all exits to itself with weight 0
        dummy_node = self.add_dummy_node(exits)

        ### Run forward Dijkstra's - O(E log V)
        # forward_travel_time = time taken from start to every other vertex in the shortest path
        # forward_previous_vertex_id = vertex id before the current vertex in the shortest path
        forward_travel_time, forward_previous_vertex_id = self.dijkstra(start)

        ### Reverse all edges - O(2E)
        self.reverse_edges_direction()
        self.reset_graph()

        ### Run backward's Dijkstra's - O(E log V)
        backward_travel_time, backward_previous_vertex_id = self.dijkstra(
            dummy_node.id
        )  # Run dijkstra on the last node, to get the shortest path from the last node to every other node

        ### Get the path with the shortest travel time, take into account the time to fight a monster to get a key - O(V)
        min_time = math.inf
        key_chosen = None

        for key_tuple in self.keys:
            key_location_index = key_tuple[0]
            time_to_get_key = key_tuple[1]
            total_time = (
                forward_travel_time[key_location_index]  # time to get from start to key
                + backward_travel_time[
                    key_location_index
                ]  # time to get from exit to key
                + time_to_get_key
            )
            # update min_time and key_chosen
            if total_time < min_time:
                key_chosen = key_location_index
                min_time = total_time

        # Exit early if no path exists
        if min_time == math.inf or key_chosen == None:
            return None

        # Get path results from start to the chosen key - O(V)
        forward_path = self.path_tracker(
            self.vertex_list[start],
            self.vertex_list[key_chosen],
            forward_travel_time,
            forward_previous_vertex_id,
        )
        forward_path.pop()  # Remove the chosen key to avoid duplicates

        # Get path results from exit to the chosen key - O(V)
        backward_path = self.path_tracker(
            self.vertex_list[dummy_node.id],
            self.vertex_list[key_chosen],
            backward_travel_time,
            backward_previous_vertex_id,
        )
        self.reverse_list(
            backward_path
        )  # Reverse backward path so it is from the chosen key to exit - O(V)

        # Combine paths and distance to get results
        combined_path = forward_path + backward_path
        if combined_path[-1] == dummy_node.id:
            combined_path.pop()
        results = (min_time, combined_path)

        # Bring everything back to normal state - O(2E)
        self.remove_dummy_node(
            exits
        )  # remove_dummy_node to state before adding dummy node
        self.reverse_edges_direction()
        self.reset_graph()

        return results

    def add_dummy_node(self, exits: list):
        """
        Function description: Add a dummy node to the graph, that connects all exits to itself with weight 0
        :Precondition: The Graph has been initialised
        :Postcondition: The dummy node is added to the graph
        :Input:
            exits: A list of numbers indicating the index of the exit vertices
        :Return:
            dummy_node: The dummy node Vertex object
        :Time complexity:
            Worst: O(E) - loop through exits list
            Best: same as worst case
        :Space complexity:
            Input: O(V) - the exits list
            Auxiliary: O(E + V) - the dummy node and the connections to it
        """
        # Add a dummy node(Vertex object) to the graph
        self.max_vertex_id += 1
        dummy_node = Vertex(self.max_vertex_id)
        self.vertex_list.append(dummy_node)

        # Connect every exits to the dummy node with weight 0 - O(E)
        for exit_node in exits:
            self.edge_list.append(
                Edge(self.vertex_list[exit_node], self.vertex_list[-1], 0)
            )
            # Add connection to dummy node
            self.vertex_list[exit_node].connections.append(self.edge_list[-1])
        return dummy_node

    def dijkstra(self, start: int):
        """
        Function description: Find the shortest path from start to every other vertex
        Approach description: Use Dijkstra's algorithm with MinHeap, discover every vertex and update the travel time and previous vertex id if a shorter path is found
        :Precondition: The Graph has been initialised
        :Input:
            start: start vertex's id
        :Return:
            tuple containing:
                - travel time: list of travel time to every other vertex from start
                - previous: list of previous vertex id to every other vertex from start
        :Time complexity:
            Worst: O(E log V) - as E <= V^2, log V is the complexity of dealing Vertex objects in the MinHeap
            Best: same as worst case
        :Space complexity:
            Input: O(1)
            Auxiliary: O(E + V) - the MinHeap and the travel time and previous lists
        """
        start_vertex = self.vertex_list[start]  # get start vertex

        ### Initialise lists and MinHeap - O(V)
        travel_time = [math.inf for _ in range(len(self.vertex_list))]
        travel_time[start_vertex.id] = 0

        previous = [0 for _ in range(len(self.vertex_list))]

        discovered = MinHeap()  # MinHeap stores tuple of (travel_time, vertex id)
        discovered.add(0, start_vertex.id)

        ### Run Dijkstra's - O(V^2 log V) = O(E log V)
        while discovered.length > 0:  # Loop until all vertices are discovered - O(V)
            [u_min_time, u] = discovered.get_min()  # O(log V)

            # If the vertex is already discovered, skip it
            if travel_time[u] <= u_min_time:
                connected_edges = self.vertex_list[
                    u
                ].connections  # bring in all the edges connected to the vertex

                for edge in connected_edges:  # O(V)
                    v = edge.v.id
                    w = edge.w

                    # If a shorter path is found, update travel_time and previous
                    if travel_time[v] > travel_time[u] + w:
                        travel_time[v] = travel_time[u] + w
                        previous[v] = u

                        # Add the updated vertex to the MinHeap
                        discovered.add(travel_time[v], v)

        return travel_time, previous

    def reverse_edges_direction(self):
        """
        Function description: Reverse all edges in the graph. i.e. if there is an edge from u to v, it will be reversed to v to u
        :Precondition: The Graph has been built
        :Postcondition: All edges in the graph are reversed
        :Input: None
        :Return: None
        :Time complexity:
            Worst: O(E) - loop through edge_list
            Best: same as worst case
        :Space complexity:
            Input: O(1)
            Auxiliary: O(1)
        """
        for edge in self.edge_list:  # O(E)
            edge.u, edge.v = edge.v, edge.u

    def reset_graph(self):
        """
        Function description: Reset the graph
        Approach description: Create a new vertex list, iterate over every edge in edge_list, reassign edge to new graph's vertices,
        add edge to source vertex's connection list, update vertex_list with new_vertex_list
        :Preconditions: The Graph has been initialised
        :Postconditions: The Graph is rebuilt
        :Input: None
        :Return: None
        :Time complexity:
            Worst: O(V + E) - loop through vertex_list and edge_list
            Best: same as worst case
        :Space complexity:
            Input: O(1)
            Auxiliary: O(V + E) - the new_vertex_list and the connections to it
        """
        # Create new vertex list - O(V)
        new_vertex_list = [Vertex(i) for i in range(self.max_vertex_id + 1)]

        # Reassign edges to new graph's vertices - O(E)
        for edge in self.edge_list:
            from_vertex_id = edge.u.id
            to_vertex_id = edge.v.id

            edge.u = new_vertex_list[from_vertex_id]
            edge.v = new_vertex_list[to_vertex_id]

            # Add edge to from_vertex's connection list
            edge.u.connections.append(edge)

        # Update vertex_list with new_vertex_list
        self.vertex_list = new_vertex_list

    def remove_dummy_node(self, exits: list):
        """
        Function description: Reset the graph to the state before adding the dummy node
        :Precondition: The Graph has been initialised
        :Postcondition: The dummy node is removed from the graph
        :Input:
            exits: A list of numbers indicating the index of the exit vertices
        :Return: None
        :Time complexity:
            Worst: O(E) - loop through the exits list
            Best: same as worst case
        :Space complexity:
            Input: O(V) - the exits list
            Auxiliary: O(E + V) - the dummy node and the connections to it
        """
        self.max_vertex_id -= 1
        self.vertex_list.pop()
        # Pop the connections of every exit node to the dummy node - O(E)
        for _ in exits:
            self.edge_list.pop()

    def path_tracker(self, from_vertex, to_vertex, travel_time, previous):
        """
        Function description: Trace the path from u to v based on the computed travel_time and previous lists
        :Precondition: travel_time and previous are computed and passed in
        :Input:
            from_vertex: vertex object of u
            to_vertex: vertex object of v
            travel_time: list of travel time to every other vertex from start
            previous: list of previous vertex id to every other vertex from start
        :Return: list of vertices id to go through to get to the exit
        :Time complexity:
            Worst: O(V)
            Best: same as worst case
        :Space complexity:
            Input: O(V) - the travel_time and previous lists
            Auxiliary: O(V) - the path list
        """
        total_travel_time = travel_time[to_vertex.id]  # Get the total travel time

        if total_travel_time == math.inf:  # If no path exists
            return [[], -1]

        current = to_vertex.id  # Start from the to_vertex
        path = []

        # Trace backwards until we reach the from_vertex - O(V)
        while current != from_vertex.id:
            path.insert(0, current)
            current = previous[current]

        path.insert(0, from_vertex.id)  # Insert the from_vertex to the path

        return path

    def reverse_list(self, list):
        """
        Function description: Reverse all elements in a given list
        :Precondition: None
        :Postcondition: The list is reversed
        :Input:
            list: The list to reverse
        :Return: None
        :Time complexity:
            Worst: O(N), where N is the length of the list
            Best: same as worst case
        :Space complexity:
            Input: O(N), where N is the length of the list
            Auxiliary: O(1)
        """
        for i in range(len(list) // 2):  # O(N/2)
            list[i], list[len(list) - 1 - i] = (
                list[len(list) - 1 - i],
                list[i],
            )  # swap the first and last element, all swapped when reach the middle
