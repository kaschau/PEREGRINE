from lxml import etree
from copy import deepcopy


class gridMetaData:
    def __init__(self, precision, lump):
        self.metaType = "grid"
        self.lump = lump
        self.precision = precision
        self.outputName = "g.xmf"

        # This is the main xdmf object
        self.tree = etree.Element("Xdmf")
        self.tree.set("Version", "2")

        self.domainElem = etree.SubElement(self.tree, "Domain")
        self.gridElem = etree.SubElement(self.domainElem, "Grid")
        self.gridElem.set("Name", "PEREGRINE Output")
        self.gridElem.set("GridType", "Collection")
        self.gridElem.set("CollectionType", "Spatial")

        # This is a template of an individual block
        self.blockTemplate = etree.Element("Grid")
        self.blockTemplate.set("Name", "B#Here")

        topologyElem = etree.SubElement(self.blockTemplate, "Topology")
        topologyElem.set("TopologyType", "3DSMesh")
        topologyElem.set("NumberOfElements", "Num Elem Here")
        geometryElem = etree.SubElement(self.blockTemplate, "Geometry")
        geometryElem.set("GeometryType", "X_Y_Z")

        dataXElem = etree.SubElement(geometryElem, "DataItem")
        dataXElem.set("NumberType", "Float")
        dataXElem.set("Dimensions", "XYZ Dims Here")
        dataXElem.set("Precision", "8" if precision == "double" else "4")
        dataXElem.set("Format", "HDF")

        dataXElem.text = "gridFile location:/coordinate/x"
        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = "gridFile location:/coordinates/y"

        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = "gridFile location:/coordinates/z"

    def saveXdmf(self, path="./"):
        et = etree.ElementTree(self.tree)
        saveFile = f"{path}/{self.outputName}"
        et.write(saveFile, pretty_print=True, encoding="UTF-8", xml_declaration=True)

    def addBlockElem(self, nblki, ni, nj, nk, ng):

        blockElem = deepcopy(self.blockTemplate)
        blockElem.set("Name", f"B{nblki:06d}")
        topo = blockElem.find("Topology")
        topo.set("NumberOfElements", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")

        for coord, i in zip(["x", "y", "z"], [0, 1, 2]):
            X = blockElem.find("Geometry")[i]
            X.set("Dimensions", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")
            X.text = self.getGridFileH5Location(coord, nblki)

        self.gridElem.append(deepcopy(blockElem))

    def getGridFileH5Location(self, coord, nblki):
        if self.metaType == "grid":
            gridPath = "."
        else:
            gridPath = self.gridPath

        if self.lump:
            return f"{gridPath}/g.h5:/coordinates_{nblki:06d}/{coord}"
        else:
            return f"{gridPath}/g.{nblki:06d}.h5:/coordinates_{nblki:06d}/{coord}"

    def getGridFileName(self, coord, nblki):
        if self.metaType == "grid":
            gridPath = "."
        else:
            gridPath = self.gridPath

        if self.lump:
            return f"{gridPath}/g.h5"
        else:
            return f"{gridPath}/g.{nblki:06d}.h5"


class restartMetaData(gridMetaData):
    def __init__(self, gridPath, precision, animate, lump, nrt=0, tme=0.0):
        super().__init__(precision, lump)
        self.metaType = "restart"

        self.animate = animate
        self.gridPath = gridPath
        if self.animate:
            self.outputName = f"q.{nrt:08d}.xmf"
        else:
            self.outputName = "q.xmf"

        self.timeElem = etree.SubElement(self.blockTemplate, "Time")
        self.timeElem.set("Value", str(tme))

        # Scalar attribute template
        self.scalarAttributeTemplate = etree.Element("Attribute")
        self.scalarAttributeTemplate.set("Name", "var name here")
        self.scalarAttributeTemplate.set("ScalarAttributeType", "Scalar")
        self.scalarAttributeTemplate.set("Center", "Cell")

        # Vector attribute template
        self.vectorAttributeTemplate = etree.Element("Attribute")
        self.vectorAttributeTemplate.set("Name", "vector name")
        self.vectorAttributeTemplate.set("AttributeType", "Vector")
        self.vectorAttributeTemplate.set("Center", "Cell")
        function = etree.SubElement(self.vectorAttributeTemplate, "DataItem")
        function.set("ItemType", "Function")
        function.set("Function", "JOIN($0, $1, $2)")
        function.set("Dimensions", "Dimension here 3")

        # Data Item template
        self.dataItemTemplate = etree.Element("DataItem")
        self.dataItemTemplate.set("NumberType", "Float")
        self.dataItemTemplate.set("Dimensions", "var dim nums here")
        self.dataItemTemplate.set("Precision", "8" if precision == "double" else "4")
        self.dataItemTemplate.set("Format", "HDF")
        self.dataItemTemplate.text = "resultFile location:/results/"

    def addBlockElem(self, nblki, ni, nj, nk, ng):

        blockElem = deepcopy(self.blockTemplate)
        blockElem.set("Name", f"B{nblki:06d}")
        topo = blockElem.find("Topology")
        topo.set("NumberOfElements", f"{nk} {nj} {ni}")

        for coord, i in zip(["x", "y", "z"], [0, 1, 2]):
            X = blockElem.find("Geometry")[i]
            X.set("Dimensions", f"{nk} {nj} {ni}")
            X.text = self.getGridFileH5Location(coord, nblki)

        self.gridElem.append(deepcopy(blockElem))

        return self.gridElem[-1]

    def getVarFileH5Location(self, varName, nblki, nrt):
        if self.lump:
            if self.animate:
                return f"q.{nrt:08d}.h5:/results_{nblki:06d}/{varName}"
            else:
                return f"q.h5:/results_{nblki:06d}/{varName}"
        else:
            if self.animate:
                return f"q.{nrt:08d}.{nblki:06d}.h5:/results_{nblki:06d}/{varName}"
            else:
                return f"q.{nblki:06d}.h5:/results_{nblki:06d}/{varName}"

    def getVarFileName(self, nblki, nrt):
        if self.lump:
            if self.animate:
                return f"q.{nrt:08d}.h5"
            else:
                return "q.h5"
        else:
            if self.animate:
                return f"q.{nrt:08d}.{nblki:06d}.h5"
            else:
                return f"q.{nblki:06d}.h5"

    def addScalarToBlockElem(self, blockElem, varName, nblki, nrt, ni, nj, nk):

        attributeElem = deepcopy(self.scalarAttributeTemplate)
        attributeElem.set("Name", varName)

        dataItemElem = deepcopy(self.dataItemTemplate)
        dataItemElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1}")
        dataItemElem.text = self.getVarH5FileLocation(varName, nblki, nrt)

        attributeElem.append(dataItemElem)
        blockElem.append(attributeElem)

    def addVectorToBlockElem(
        self, blockElem, vectorName, varNames, nblki, nrt, ni, nj, nk
    ):

        attributeElem = deepcopy(self.vectorAttributeTemplate)
        attributeElem.set("Name", vectorName)
        functionElem = attributeElem.find("DataItem")
        functionElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1} 3")
        for varName in varNames:
            dataItemElem = deepcopy(self.dataItemTemplate)
            dataItemElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1}")
            dataItemElem.text = self.getVarH5FileLocation(varName, nblki, nrt)

            functionElem.append(dataItemElem)

        blockElem.append(attributeElem)


class arbitraryMetaData(restartMetaData):
    def __init__(self, arrayName, precision, animate, lump, nrt=0, tme=0.0):
        super().__init__(precision, lump)
        self.metaType = "arbitrary"
        if self.animate:
            self.outputName = f"{arrayName}.{nrt:08d}.xmf"
        else:
            self.outputName = f"{arrayName}.xmf"
        self.arrayName = arrayName

    def getArrayNameIndicies(self, arrayName, speciesNames):
        ns = len(speciesNames)

        if arrayName == "q":
            raise ValueError("Use the restart writer to write out q.")

        elif arrayName == "Q":
            d = {
                "scalars": {
                    "names": ["rho", "E"] + [f"rho{i}" for i in speciesNames[0::-1]],
                    "indicies": [0, 4] + [5 + i for i in range(ns - 1)],
                },
                "vectors": {"names": ["Momentum"]},
                "lables": [["rhou", "rhov", "rhow"]],
                "indicies": [[1, 2, 3]],
            }

        return d